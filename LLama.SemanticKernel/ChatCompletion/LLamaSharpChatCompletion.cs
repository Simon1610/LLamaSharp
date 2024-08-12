using LLama;
using LLama.Abstractions;
using LLamaSharp.SemanticKernel.Connectors.Llama;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using static LLama.InteractiveExecutor;
using static LLama.LLamaTransforms;

namespace LLamaSharp.SemanticKernel.ChatCompletion;

/// <summary>
/// LLamaSharp ChatCompletion
/// </summary>
public sealed class LLamaSharpChatCompletion : IChatCompletionService
{
    private readonly ILLamaExecutor _model;
    private readonly LLamaSharpPromptExecutionSettings _defaultRequestSettings;
    private readonly IHistoryTransform _historyTransform;
    private readonly ITextStreamTransform _outputTransform;

    private readonly Dictionary<string, object?> _attributes = new();
    private readonly bool _isStatefulExecutor;

    public IReadOnlyDictionary<string, object?> Attributes => _attributes;

    private static LLamaSharpPromptExecutionSettings GetDefaultSettings()
    {
        return new LLamaSharpPromptExecutionSettings
        {
            MaxTokens = 256,
            Temperature = 0,
            TopP = 0,
            StopSequences = new List<string>()
        };
    }

    public LLamaSharpChatCompletion(ILLamaExecutor model,
        LLamaSharpPromptExecutionSettings? defaultRequestSettings = default,
        IHistoryTransform? historyTransform = null,
        ITextStreamTransform? outputTransform = null)
    {
        _model = model;
        _isStatefulExecutor = _model is StatefulExecutorBase;
        _defaultRequestSettings = defaultRequestSettings ?? GetDefaultSettings();
        _historyTransform = historyTransform ?? new HistoryTransform();
        _outputTransform = outputTransform ?? new KeywordTextOutputStreamTransform(new[] { $"{LLama.Common.AuthorRole.User}:",
                                                                                            $"{LLama.Common.AuthorRole.Assistant}:",
                                                                                            $"{LLama.Common.AuthorRole.System}:"});
    }

    public ChatHistory CreateNewChat(string? instructions = "")
    {
        var history = new ChatHistory();

        if (instructions != null && !string.IsNullOrEmpty(instructions))
        {
            history.AddSystemMessage(instructions);
        }

        return history;
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var settings = executionSettings != null
           ? LLamaSharpPromptExecutionSettings.FromRequestSettings(executionSettings)
           : _defaultRequestSettings;

        var autoInvoke = kernel is not null && settings.AutoInvoke == true;

        if (!autoInvoke || kernel is null)
        {
            var prompt = _historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());
            var result = _model.InferAsync(prompt, settings.ToLLamaSharpInferenceParams(), cancellationToken);

            var output = _outputTransform.TransformAsync(result);

            var sb = new StringBuilder();
            await foreach (var token in output)
            {
                sb.Append(token);
            }

            return new List<ChatMessageContent> { new(AuthorRole.Assistant, sb.ToString()) }.AsReadOnly();
        }

        var promptFunctionCall = (_historyTransform as HistoryTransform)?.HistoryToTextFC(chatHistory.ToLLamaSharpChatHistory());
        var resultFunctionCall = _model.InferAsync(promptFunctionCall, settings.ToLLamaSharpInferenceParams(), cancellationToken);
        var outputFunctionCall = _outputTransform.TransformAsync(resultFunctionCall);
        var sbFunctionCall = new StringBuilder();

        await foreach (var token in outputFunctionCall)
        {
            sbFunctionCall.Append(token);
        }

        // Add {" to the start of the string as an hack for better results.
        var functionCallString = sbFunctionCall.ToString();
        if (!functionCallString.StartsWith("{"))
            functionCallString = "{\"" + functionCallString;

        var historyToAppend = new List<ChatMessageContent> { new(new AuthorRole("FunctionCall"), functionCallString)};

        try
        {
            var parsedJson = JsonSerializer.Deserialize<LLamaFunctionCall>(functionCallString);
            if (parsedJson == null) throw new NullReferenceException();

            KernelFunction? function;
            kernel.Plugins.TryGetFunction(pluginName: null, functionName: parsedJson.name, out function);
            if (function == null) throw new NullReferenceException();

            var arguments = new KernelArguments(parsedJson.parameters);

            var functionResult = await kernel.InvokeAsync(function, arguments, cancellationToken);
            historyToAppend.Add(new ChatMessageContent(new AuthorRole("FunctionResult"), functionResult.GetValue<object>().ToString()));
            chatHistory.AddRange(historyToAppend);
            var promptFunctionResult = (_historyTransform as HistoryTransform)?.HistoryToTextFC(chatHistory.ToLLamaSharpChatHistory());
            var resultFunctionResult = _model.InferAsync(promptFunctionResult, settings.ToLLamaSharpInferenceParams(), cancellationToken);
            var outputFunctionResult = _outputTransform.TransformAsync(resultFunctionResult);

            var sbFunctionResult = new StringBuilder();
            await foreach (var token in outputFunctionResult)
            {
                sbFunctionResult.Append(token);
            }

            return new List<ChatMessageContent> { new(AuthorRole.Assistant, sbFunctionResult.ToString()) }.AsReadOnly();
        }
        catch (Exception e)
        {
            Debug.WriteLine(e);
            var lastChatMessage = historyToAppend.LastOrDefault();
            Debug.WriteLine(lastChatMessage.Content);
            switch (lastChatMessage.Role.ToString())
            {
                case "FunctionCall":
                    historyToAppend.Remove(historyToAppend.LastOrDefault());
                    break;
                case "FunctionResult":
                    historyToAppend.RemoveRange(historyToAppend.Count - 2, 2);
                    break;
            }

            var prompt = (_historyTransform as HistoryTransform)?.HistoryToTextFC(chatHistory.ToLLamaSharpChatHistory());
            var result = _model.InferAsync(prompt, settings.ToLLamaSharpInferenceParams(), cancellationToken);
            var output = _outputTransform.TransformAsync(result);

            var sb = new StringBuilder();
            await foreach (var token in output)
            {
                sb.Append(token);
            }

            historyToAppend.Add(new ChatMessageContent(AuthorRole.Assistant, sb.ToString()));
            return historyToAppend.AsReadOnly();
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(ChatHistory chatHistory,
                                                                                                    PromptExecutionSettings? executionSettings = null,
                                                                                                    Kernel? kernel = null,
                                                                                                    [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var settings = executionSettings != null
          ? LLamaSharpPromptExecutionSettings.FromRequestSettings(executionSettings)
          : _defaultRequestSettings;

        var autoInvoke = kernel is not null && settings.AutoInvoke == true;

        if(!autoInvoke || kernel is null)
        {
            var prompt = _getFormattedPrompt(chatHistory);
            var result = _model.InferAsync(prompt, settings.ToLLamaSharpInferenceParams(), cancellationToken);
            var output = _outputTransform.TransformAsync(result);

            await foreach (var token in output)
            {
                yield return new StreamingChatMessageContent(AuthorRole.Assistant, token);
            }
        }
        else
        {
            var promptFunctionCall = (_historyTransform as HistoryTransform)?.HistoryToTextFC(chatHistory.ToLLamaSharpChatHistory());
            var resultFunctionCall = _model.InferAsync(promptFunctionCall, settings.ToLLamaSharpInferenceParams(), cancellationToken);
            var outputFunctionCall = _outputTransform.TransformAsync(resultFunctionCall);
            var sbFunctionCall = new StringBuilder();

            await foreach (var token in outputFunctionCall)
            {
                sbFunctionCall.Append(token);
            }

            sbFunctionCall.Replace("System:", null);
            sbFunctionCall.Replace("User:", null);
            sbFunctionCall.Replace("Assistant:", null);

            // Add {" to the start of the string as an hack for better results.
            var functionCallString = sbFunctionCall.ToString();
            if (functionCallString.Contains("Answer:")) { functionCallString = functionCallString.Remove(functionCallString.IndexOf("Answer:")); }else if(functionCallString.Contains("Result:")) { functionCallString = functionCallString.Remove(functionCallString.IndexOf("Result:")); };
            if (!functionCallString.StartsWith("{"))
            {
                functionCallString = "{\"" + functionCallString;
            }else if (!functionCallString.StartsWith("{\""))
            {
                functionCallString = "{\"" + functionCallString.Remove(0);
            }

            var historyToAppend = new List<ChatMessageContent>
                { new(new AuthorRole("FunctionCall"), functionCallString)};

            IAsyncEnumerable<string> outputFunctionResultOrCatch;
            try
            {
                var parsedJson = JsonSerializer.Deserialize<LLamaFunctionCall>(functionCallString);
                if (parsedJson == null) throw new NullReferenceException();

                KernelFunction? function;
                kernel.Plugins.TryGetFunction(pluginName: null, functionName: parsedJson.name, out function);
                if (function is null) throw new NullReferenceException();

                var arguments = new KernelArguments(parsedJson.parameters);

                var functionResult = await kernel.InvokeAsync(function, arguments, cancellationToken);
                historyToAppend.Add(new ChatMessageContent(new AuthorRole("FunctionResult"),
                    functionResult.GetValue<object>().ToString()));
                chatHistory.AddRange(historyToAppend);
                var promptFunctionResult = _historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());
                var resultFunctionResult = _model.InferAsync(promptFunctionResult,
                    settings.ToLLamaSharpInferenceParams(),
                    cancellationToken);
                outputFunctionResultOrCatch = _outputTransform.TransformAsync(resultFunctionResult);
            }
            catch (Exception e)
            {
                Debug.WriteLine(e);
                var lastChatMessage = historyToAppend.Last();
                Debug.WriteLine(lastChatMessage.Content);
                switch (lastChatMessage.Role.ToString())
                {
                    case "FunctionCall":
                        historyToAppend.Remove(historyToAppend.LastOrDefault());
                        break;
                    case "FunctionResult":
                        historyToAppend.RemoveRange(historyToAppend.Count - 2, 2);
                        break;
                }

                var prompt = _historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());
                var result = _model.InferAsync(prompt, settings.ToLLamaSharpInferenceParams(), cancellationToken);
                outputFunctionResultOrCatch = _outputTransform.TransformAsync(result);
            }

            await foreach (var token in outputFunctionResultOrCatch)
            {
                yield return new StreamingChatMessageContent(AuthorRole.Assistant, token);
            }
        }
    }

    /// <summary>
    /// Return either the entire formatted chatHistory or just the most recent message based on
    /// whether the model extends StatefulExecutorBase or not.
    /// </summary>
    /// <param name="chatHistory"></param>
    /// <returns>The formatted prompt</returns>
    string _getFormattedPrompt(ChatHistory chatHistory)
    {
        string prompt;
        if (_isStatefulExecutor)
        {
            var state = (InteractiveExecutorState)((StatefulExecutorBase)_model).GetStateData();
            if (state.IsPromptRun)
            {
                prompt = _historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());
            }
            else
            {
                ChatHistory tempHistory = new();
                tempHistory.AddUserMessage(chatHistory.Last().Content ?? "");
                prompt = _historyTransform.HistoryToText(tempHistory.ToLLamaSharpChatHistory());
            }
        }
        else
        {
            prompt = _historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());
        }

        return prompt;
    }
}

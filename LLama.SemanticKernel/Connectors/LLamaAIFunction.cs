using Microsoft.SemanticKernel;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LLamaSharp.SemanticKernel.Connectors.Llama;

public sealed class LLamaFunctionCall
{
    public string name { get; set; }

    [JsonConverter(typeof(DictionaryStringObjectConverter))]
    public Dictionary<string, object?> parameters { get; set; }
}

public class DictionaryStringObjectConverter : JsonConverter<Dictionary<string, object?>>
{
    public override Dictionary<string, object?> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException();
        }

        Dictionary<string, object?> dictionary = new Dictionary<string, object?>();
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                return dictionary;
            }

            if (reader.TokenType != JsonTokenType.PropertyName)
            {
                throw new JsonException();
            }

            string propertyName = reader.GetString();
            reader.Read();
            object? propertyValue;
            switch (reader.TokenType)
            {
                case JsonTokenType.String:
                    propertyValue = reader.GetString();
                    break;
                case JsonTokenType.Number:
                    propertyValue = reader.TryGetInt64(out long longValue) ? longValue : reader.GetDouble();
                    break;
                default:
                    throw new JsonException($"Unsupported token type: {reader.TokenType}");
            }
            dictionary[propertyName] = propertyValue;
        }

        throw new JsonException("Unclosed object at end of JSON.");
    }
    public override void Write(Utf8JsonWriter writer, Dictionary<string, object?> value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        foreach (var kvp in value)
        {
            writer.WritePropertyName(kvp.Key);
            JsonSerializer.Serialize(writer, kvp.Value, options);
        }
        writer.WriteEndObject();
    }
}
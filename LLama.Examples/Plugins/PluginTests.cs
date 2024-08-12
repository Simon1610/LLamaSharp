using Microsoft.SemanticKernel;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLama.Examples.Plugins
{
    internal class PluginTests
    {
        [KernelFunction]
        [Description("Take the square root of a number")]
        public static double Sqrt([Description("The number to take a square root of")]double number1)
        {
            Debug.WriteLine("Sqrt called");
            return Math.Sqrt(number1);
        }

        [KernelFunction]
        [Description("Add two numbers")]
        public static double Add([Description("The first number to add")]double number1,[Description("The second number to add")]double number2)
        {
            Debug.WriteLine("Add called");
            return number1 + number2;
        }

        [KernelFunction]
        [Description("Gets the summary of a text")]
        public static string Summarize([Description("The text to summarize")]string text)
        {
            Debug.WriteLine("Summarize called");
            return $"The summary of {text}";
        }
    }
}

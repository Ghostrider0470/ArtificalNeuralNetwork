using System;
using System.Collections;
using System.Collections.Generic;
using myFirstArtificalNeuralNetwork;
using Random = myFirstArtificalNeuralNetwork.Random;
public class ArtificialNeuralNetwork
{
    // NumberOfInputs is a number of Inputs coming into the Neural Network
    public int NumberOfInputs;

    // NumberOfInputs is a number of Outputs coming out the Neural Network
    public int NumberOfOutputs;

    // Number of Hidden Layers in the Neural Network
    public int NumberOfHiddenLayers;

    // Number of Neurons Per Hidden Layer
    public int NumberOfNeuronsPerHiddenLayer;

    // AlphaMultiplier is used to dictate the impact of a TrainingSet on the Network
    public double AlphaMultiplier;

    // Layers is a list of Neural Layers
    private List<Layer> Layers = new List<Layer>();

    // The constructor for the ArtificialNeuralNetwork
    public ArtificialNeuralNetwork(int numberOfInputs, int numberOfOutputs,
        int numberOfHiddenLayers, int numberOfNeuronsPerHiddenLayer, double alphaMultiplier)
    {
        // This block of code sets root variables to the values of the variables exposed in the constructor
        {
            NumberOfInputs = numberOfInputs;
            NumberOfOutputs = numberOfOutputs;
            NumberOfHiddenLayers = numberOfHiddenLayers;
            NumberOfNeuronsPerHiddenLayer = numberOfNeuronsPerHiddenLayer;
            AlphaMultiplier = alphaMultiplier;
        }

        // This If Statement checks if this is a Perceptron or a Neural Network
        if (numberOfHiddenLayers > 0)
        {   // If this is a Neural Network

            //Add the Input Layer to the list of Layers
            Layers.Add(new Layer(NumberOfNeuronsPerHiddenLayer, NumberOfInputs));

            // Iterate thru the NumberOfHiddenLayers and 
            for (int i = 0; i < NumberOfHiddenLayers; i++)
            {
                // For each iteration generate a new Layer and pass the NumberOfNeuronsPerHiddenLayer and a NumberOfInputs
                // In this case we are passing a NumberOfNeuronsPerHiddenLayer two times
                // couse the number of Inputs for the hidden Layer is equal to the NumberOfNeuronsPerHiddenLayer
                Layers.Add(new Layer(NumberOfNeuronsPerHiddenLayer, NumberOfNeuronsPerHiddenLayer));
            }
            // Finally we add the Output Layer
            Layers.Add(new Layer(NumberOfOutputs, NumberOfNeuronsPerHiddenLayer));
        }
        else
        {
            // We add the only Layer
            Layers.Add(new Layer(NumberOfOutputs, NumberOfInputs));
        }
    }

    public List<double> CalculateOutput(List<double> inputValues, List<double> desiredOutputs)
    {
        // Create a variable to store list of inputs
        List<double> inputs;

        // Create a variable to store Output list and initialize it an empty list
        List<double> outputs = new List<double>();

        // This If Statement checks of the number of input values in the input list doesn't match the NumberOfInputs
        if (inputValues.Count != NumberOfInputs)
        {
            //Debug.Log($"ERROR: inputValues parameter is incorrect size, count of contained values must be = {NumberOfInputs}");
            return outputs;
        }

        // Initialize the list of Inputs first time with the values passed in the constructor
        inputs = new List<double>(inputValues);

        // Loops around for each hidden layer and the output layer
        for (int layer = 0; layer < NumberOfHiddenLayers + 1; layer++)
        {
            //Checks to see if this isn't the first layer
            if (layer > 0)
            {
                // If it isn't the first layer overwrite the values of the Input list 
                // and initialize it with the values of the Output of the previos layer
                inputs = new List<double>(outputs);
            }
            // Reset and clear the values of the 'output' variable from the values of the previos layers output
            outputs.Clear();

            // Loop thru all the neurons on the current layer in the Layers List
            for (int neuron = 0; neuron < Layers[layer].NumberOfNeurons; neuron++)
            {
                // 'dotProduct' is the product of each Input * Weight - Bias
                // here we set it to 0 
                double dotProduct = 0;

                // We are accessing the input list in the current neuron
                // in the list of neuron in the current layer in the list of layers
                // then we execute .Clear() to wipe that list
                // the reason for this is couse this is the data from a previous layer
                Layers[layer].Neurons[neuron].Inputs.Clear();

                // Loop thru all the current neuron in the list of neurons in the current layer in the layer list
                // for the 'NumberOfInputs'
                for (int input = 0; input < Layers[layer].Neurons[neuron].NumberOfInputs; input++)
                {
                    // For this current input we add the input from the list of inputs 
                    // previously overwritten with the list of outputs from the previous layer
                    // we add this list to the current neuron in the list of neurons in the current layer in the list of layers
                    Layers[layer].Neurons[neuron].Inputs.Add(inputs[input]);

                    // Here we partially calculate dotProduct for this neuron 
                    // we do so by multiplying current neurons inputs with their weights
                    dotProduct += Layers[layer].Neurons[neuron].Weights[input] * inputs[input];
                }

                // Here we finish the 'dotProduct' calculation by subtracting the current neurons Bias from the final sum;
                dotProduct -= Layers[layer].Neurons[neuron].Bias;

                // Here we pass the calculated dotProduct to the Activation Function which in turn does a chosen mathematical function on it
                // and returns the output for this neuron
                Layers[layer].Neurons[neuron].Output = ActivationFunction(dotProduct);

                // Here we add the calculated value of the neuron to the outputs list
                outputs.Add(Layers[layer].Neurons[neuron].Output);
            }

        }




        return outputs;
    }

    void UpdateWeights(List<double> outputs, List<double> desiredOutputs)
    {
        // Total Error of the whole neural network
        double error;

        // Looping thru the Hidden layers and the output backwards
        for (int layer = NumberOfHiddenLayers + 1; layer >= 0; layer--)
        {
            // Looping thru the neurons in the current layer
            for (int neuron = 0; neuron < Layers[layer].NumberOfNeurons; neuron++)
            {
                // If this is the Output layer
                if (layer == NumberOfHiddenLayers + 1)
                {
                    // Calculate the total error in the last layer
                    error = desiredOutputs[neuron] - outputs[neuron];

                    // Calculate the error gradient of the last layer using the Delta Rule: en.wikipedia.org/wiki/Delta_rule
                    Layers[layer].Neurons[neuron].ErrorGradient = outputs[layer] * (1 - outputs[layer]) * error;
                }
                // If this is not the output layer
                else
                {

                    // Here we partially calculate the ErrorGradient by Delta Rule but the error multiplication will be done later
                    Layers[layer].Neurons[neuron].ErrorGradient = Layers[layer].Neurons[neuron].Output * (1 - Layers[layer].Neurons[neuron].Output);

                    // Here we prepare the variable that is gonna be used to multiply the partially calculated ErrorGradient to get the final result
                    // of the ErrorGradient
                    double errorGradientSum = 0;

                    // Here we prepare a variable to store the previous layer
                    int previousLayer = layer + 1;

                    // Here we loop (iterate) thru the Neurons of the previous layer that has the ErrorGradient calculated
                    for (int previousNeurons = 0; previousNeurons < Layers[layer + 1].NumberOfNeurons; previousNeurons++)
                    {
                        // Here we get the errorGradientSum by multiplying the ErrorGradient of the previous layer with the
                        // Weight of the for each neuron from the previous layer connected to the current neuron on the current layer
                        errorGradientSum += Layers[previousLayer].Neurons[previousNeurons].ErrorGradient * Layers[previousLayer].Neurons[previousNeurons].Weights[neuron];
                    }
                    // Here we finally calculate the current layers errorGradient by multiplying it with the errorGradientSum
                    Layers[layer].Neurons[neuron].ErrorGradient *= errorGradientSum;
                }
                // Looping thru the Weights of the current neuron
                for (int weight = 0; weight < Layers[layer].Neurons[neuron].NumberOfInputs; weight++)
                {
                    // This block of code executes only if this is the output layer
                    if (layer == NumberOfHiddenLayers + 1)
                    {
                        // Calculate the total error in the last layer
                        error = desiredOutputs[neuron] - outputs[neuron];

                        // Recalculate the weight of a neuron by adding the
                        // result of the multiplication of the AlphaMultiplier, error value and the value of the input associated with that specific weight
                        Layers[layer].Neurons[neuron].Weights[weight] +=
                            AlphaMultiplier * Layers[layer].Neurons[neuron].Inputs[weight] * error;
                    }
                    // This block of code is executed if its any other layer other than the output
                    else
                    {
                        // Recalculate the weight of a neuron by adding the
                        // result of the multiplication of the AlphaMultiplier, the value of the input associated with that specific weight and the errorGradient
                        Layers[layer].Neurons[neuron].Weights[weight] +=
                            AlphaMultiplier * Layers[layer].Neurons[neuron].Inputs[weight] * Layers[layer].Neurons[neuron].ErrorGradient;
                    }
                }
                // Recalculating the bias value for each neuron by multiplying the AlphaMultiplier and the ErrorGradient and then subtracting the value
                Layers[layer].Neurons[neuron].Bias -= AlphaMultiplier * Layers[layer].Neurons[neuron].ErrorGradient;
            }
        }
    }
    // for full list of activation functions
    // see en.wikipedia.org/wiki/Activation_function
    internal double ActivationFunction(double value)
    {
        return Sigmoid(value);
    }

    #region Matematicke Funkcije
    internal double Step(double value)// AKA (Binary step)
    {
        if (value < 0) return 0;
        else return 1;
    }
    internal double Sigmoid(double value)// AKA (Logistic softstep)
    {
        double k = (double)System.Math.Exp(value);
        return k / (1.0f + k);
    }
    #endregion
}

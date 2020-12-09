using System;
using System.Collections;
using System.Collections.Generic;
using Random = myFirstArtificalNeuralNetwork.Random;
// Neuron = lowest level of data manipulation in a neural network
// it is a  mathematical function that takes the input and calculates it into an output 
public class Neuron
{
    public int NumberOfInputs;

    // Bias is a multiplier for each neuron that allows tweaking of the output value
    public double Bias;

    // Input values that go into the function which can be passed in the input layer or from a previos layer
    public List<double> Inputs = new List<double>();

    // Weight is a multiplier for a specific Input which dictates the value of the Output
    public List<double> Weights = new List<double>();

    // Output is a the result of the mathematical function 
    public double Output;

    // ErrorGradiant is the Neurons share in the TotalError of the Neural Network
    public double ErrorGradient;

    // Constructor for a Neuron 
    public Neuron(int numberOfInputs)
    {

        // Assigning a random value to the Bias in a range of -1 to 1 
        Bias = Random.Range(-1.0f, 1.0f);

        // Sets Number of Inputs to the number assigned to the variable exposed in the constructor
        NumberOfInputs = numberOfInputs;

        // Iterates thru the number of inputs which equals the number of weights 
        for (int i = 0; i < NumberOfInputs; i++)
        {   // for each iteration (for Weight)

            // Assign a random value to the Weight in a range of -1 to 1 
            Weights.Add(Random.Range(-1.0f, 1.0f));
        }
    }
}

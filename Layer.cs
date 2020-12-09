using System.Collections;
using System.Collections.Generic;


// Layer is a group of a certain number of Neurons
public class Layer
{
    // Number of Neurons in a Layer
    public int NumberOfNeurons;

    // Neurons is a list of Neuron objects in a Layer
    public List<Neuron> Neurons = new List<Neuron>();

    // Constructor for a Layer
    public Layer(int numberOfNeurons, int numberOfNeuronInputs)
    {
        // Sets the NumberOfNeurons to the value of numberOfNeurons variable exposed in the constructor
        NumberOfNeurons = numberOfNeurons;

        // Iterates thru NumberOfNeurons
        for (int i = 0; i < NumberOfNeurons; i++)
        {
            // Creates a new Neuron with a specific number of Inputs and adds it to a list of Neurons
            Neurons.Add(new Neuron(numberOfNeuronInputs));
        }
    }
}

SHELL = /bin/bash
#Compiler
COMPILER = g++ -g -std=c++11 -Wall



all: layer 

layer: main.o NeuronLayer.o Neuron.o NeuronLayer.h Neuron.h
	$(COMPILER) $^ -o $@ 

Neuron.o: Neuron.cpp Neuron.h 
	$(COMPILER) -c $<

NeuronLayer.o: NeuronLayer.cpp NeuronLayer.h Neuron.h
	$(COMPILER) -c $< 

main.o: main.cpp NeuronLayer.h Neuron.h
	$(COMPILER) -c $<






#clean up
clean:
	rm layer *\.o

#git add files
gitadd:
	git add NeuronLayer.cpp NeuronLayer.h main.cpp Neuron.h Neuron.cpp makefile

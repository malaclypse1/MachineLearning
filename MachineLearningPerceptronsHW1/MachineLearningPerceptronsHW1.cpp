/*===========================================================================
Program:	MachineLearningPerceptronsHW1
Author:		Troy Routley
Date:		01/22/2017
Dev Env:	Visual Studio 2015
Description: Simple machine learning program taking input of 60,000 training
samples from "mnist_train.csv" and 10,000 testing samples from "mnist_test.csv
and training a single-layer neural network to identify hand-written digits.

Uses the eigen library for matrix manipulation: eigen.tuxfamily.org
=============================================================================*/

#include "stdafx.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <windows.h> //QueryPerformanceFrequency & QueryPerformanceCounter
#include "MachineLearningPerceptronsHW1.h"

using namespace std;
using namespace Eigen;

#define TRAINSIZE 60000
#define TESTSIZE 10000
#define SEED 5
#define MAX_EPOCHS 70

struct Data
{
	short t;						//target digit
	Matrix<float, 785,1> pixels;	//row vector of pixels
};

class WeightMatrix
{
private:
	Matrix<float, 10, 785> w;		//weight matrix
	Matrix<int, 10, 10> confusion;	//confusion matrix
									//populated during ComputeAccuracy

public:
	void Initialize(unsigned int seed);
	float ComputeAccuracy(Data data[], unsigned short dataSize);
	void TrainSample(Data datum, float learningRate);
	void PrintConfusion();
	void WriteConfusionToFile(float learningRate);
};

int readDataFromFile(Data data[], string filename, unsigned short quant);
int writeDataToFile(float lr, int epoch, float trainAcc, float testAcc);

int main()
{
	WeightMatrix percep;
	Data *train = new Data[TRAINSIZE];
	Data *test = new Data[TESTSIZE];

	if (readDataFromFile(train, "mnist_train.csv", TRAINSIZE)) return 1;
	//if (readDataFromFile(test, "mnist_test.csv", TESTSIZE)) return 1;

	/*
	//run 3 experiments, with learning rates of 0.001, 0.01, and 0.1
	for (float learningRate = 0.001f; learningRate < 0.2f; learningRate *= 10.0f) {

		//initialize perceptron weights randomly to 0.5 or -0.5
		percep.Initialize(SEED);

		cout << "Learning Rate: " << learningRate << endl;

		//repeat until accuracy stops improving or MAX_EPOCHS epochs reached
		float trainAcc, testAcc;	//accuracy results for training and test data
		float lastAcc = 0.0f;		//last epoch training data accuracy
		bool done = false;
		int epoch = 0;

		while (!done)
		{
			cout << "Epoch " << epoch << endl;
			// skip training for epoch 0 - go right to computing initial accuracy
			if (epoch > 0) {
				cout << "training epoch " << epoch << endl;
				//cycle through training data changing weights after processing each sample
				for (int sample = 0; sample < TRAINSIZE; sample++) {
					percep.TrainSample(train[sample], learningRate);
				}
			}
			//compute accuracy for that epoch
			trainAcc = percep.ComputeAccuracy(train, TRAINSIZE);
			testAcc = percep.ComputeAccuracy(test, TESTSIZE);
			cout << "Training Data Accuracy: " << trainAcc
				<< " Test Data Accuracy: " << testAcc << endl;
			writeDataToFile(learningRate, epoch, trainAcc, testAcc);

			epoch++;
			if (abs(trainAcc - lastAcc) < 0.005f || epoch > MAX_EPOCHS) done = true;
			lastAcc = trainAcc;
		}

		percep.PrintConfusion();
		percep.WriteConfusionToFile(learningRate);
	}
	*/

	cout << "Run complete\n";
	delete train;
	delete test;
	//keep console window open
	cin.get();

    return 0;
}
/*===========================================================================
Function:	readDataFromFile
Description: Loads training or test data from .csv files. This function is dumb
	and assumes that the .csv is correct, having no more than quant lines each
	containing 785 comma-delimited numbers.
	Performance improves about 15 fold in with release-build optimizations
Parameters:
	data[] (I/O) - reference to heap allocated Data array
	filename (I) - name of .csv file
	quant (I)	 - size of Data array/number of samples to read
=============================================================================*/
int readDataFromFile(Data data[], string filename, unsigned short quant)
{
	ifstream file;
	file.open(filename);
	int progress = quant / 60; // for visual indicator of read progress
	LARGE_INTEGER startTime;
	LARGE_INTEGER endTime;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency); //frequency of performance counter

	if (file.is_open())
	{
		QueryPerformanceCounter(&startTime);
		cout << "reading from file " << filename << endl;
		for (int datum = 0; datum < quant; datum++) {
			if (datum > 0 && datum % progress == 0)
			{
				cout << '.';
			}
			string line;
			getline(file, line);
			if (!file.good()) {
				cout << "file read error";
				break;
			}
			//parse line into values

			stringstream iss(line);

			string val;

			for (int x = 0; x < 785; x++) {
				getline(iss, val, ',');
				if (iss.fail()) {
					cout << "line parse error\n";
					break;
				}

				stringstream convertor(val);
				if (x == 0) {
					convertor >> data[datum].t;
					data[datum].pixels(0, 0) = 1.0f; // set bias
				}
				else {
					convertor >> data[datum].pixels(x, 0);
					data[datum].pixels(x, 0) /= 255.0f; //normalize 0-255 to 0-1
				}
			}
		}
		QueryPerformanceCounter(&endTime);
		cout << "\nsuccessfully read " << quant << " samples from " << filename << " in " << (endTime.QuadPart - startTime.QuadPart) / frequency.QuadPart << " seconds.\n";
	}
	else {
		cout << "file " << filename << " not found\n";
	}
	file.close();

	return(0);
}
/*===========================================================================
Function:	writeDataToFile
Description: writes accuracy data to accuracy.csv
Parameters:
	lr (I)						 - learning rate, for labelling purposes
	epoch, trainAcc, testAcc (I) - data to write
Returns 0 in successful
=============================================================================*/
int writeDataToFile(float lr, int epoch, float trainAcc, float testAcc)
{
	ofstream ofs;
	static bool firstRun = true;

	//overwrite previous runs
	if (firstRun)
	{
		ofs.open("accuracy.csv", ofstream::out | ofstream::trunc);
		firstRun = false;
	}
	else
	{
		ofs.open("accuracy.csv", ofstream::out | ofstream::app);
	}
	if (epoch == 0)
	{
		ofs << "learning rate, " << lr << endl;
		ofs << "epoch, traini acc, test acc" << endl;
	}
	ofs << epoch << ", " << trainAcc << ", " << testAcc << endl;
	ofs.close();

	return 0;
}
/*===========================================================================
Method:	Initialize
Description: initializes weight matrix with random weights, either 0.5 or -0.5
Parameters:
	seed (I)	 - seed for random numbers
=============================================================================*/
void WeightMatrix::Initialize(unsigned int seed)
{
	srand(seed);
	for (int x = 0; x < 10;  x++ )
		for (int y = 0; y < 785;  y++)
			w(x,y) = ((rand() % 2) ? -0.5f : 0.5f);
}
/*===========================================================================
Method:	ComputeAccuracy
Description: calculates accuracy of perceptron and populates confusion matrix
Parameters:
	data[] (I)	 - array of Data samples to test on
	dataSize (I) - size of data array
Returns accuracy, and overwrites confusion matrix
=============================================================================*/
float WeightMatrix::ComputeAccuracy(Data data[], unsigned short dataSize)
{
	int hits = 0;			//zero hit counter
	confusion.setZero();	//zero confusion matrix

	for (int sample = 0; sample < dataSize; sample++) {
		Matrix<float, 10, 1> result = w * data[sample].pixels;
		Matrix<float, 10, 1>::Index index;
		result.col(0).maxCoeff(&index);
		//cout << "sample: " << sample << " max: " << index 
		//	<< " expected: " << data[sample].t << endl << result << endl;
		if (index == data[sample].t) hits++;
		confusion(data[sample].t, index)++;
	}
	float accuracy = static_cast<float>(hits) / dataSize;
	//cout << "Hits: " << hits << " Sample size: " << dataSize << " Accuracy: " << accuracy << endl;
	return accuracy;
}
/*===========================================================================
Method:	TrainSample
Description: Trains perceptron on a given data sample
Parameters:
	datum (I)		 - Data struct sample to train perceptron on
	learningRate (I)
=============================================================================*/
void WeightMatrix::TrainSample(Data datum, float learningRate)
{
	//get prediction for sample
	Matrix<float, 10, 1> result = w * datum.pixels;	// w dot x
	Matrix<float, 10, 1>::Index index;				//prediction
	result.col(0).maxCoeff(&index);

	if (index != datum.t) //not a hit
	{
		//update weights
		// t = vector with '1' in correct output place, '0's elsewhere
		Matrix<float, 10, 1> t;
		t << Matrix<float, 10, 1>::Zero();
		t(datum.t) = 1;
		//cout << "t: " << t << endl;

		// y = vector with '1' where perceptron fires, '0's elsewhere
		Matrix<float, 10, 1> y;
		for (int x = 0; x < 10; x++)
		{
			y(x) = (result(x) > 0 ? 1.0f : 0.0f);
		}

		//w = w + delta w
		w += learningRate * ((t - y) * datum.pixels.transpose());
	}
}
/*===========================================================================
Method:	PrintConfusion
Description: outputs confusion matrix to cout
=============================================================================*/
void WeightMatrix::PrintConfusion()
{
	cout << confusion << endl;
}
/*===========================================================================
Method:	WriteConfusionToFile
Description: writes confusion matrix to confusion.csv file
Parameters:
	learningRate (I) - used to label confusion matrix in file
=============================================================================*/
void WeightMatrix::WriteConfusionToFile(float learningRate)
{
	ofstream ofs;
	static bool firstRun = true;
	IOFormat CommaDel(StreamPrecision, DontAlignCols, ", ", "\n", "", "", "", "");

	//overwrite previous runs
	if (firstRun)
	{
		ofs.open("confusion.csv", ofstream::out | ofstream::trunc);
		firstRun = false;
	}
	else
	{
		ofs.open("confusion.csv", ofstream::out | ofstream::app);
	}
	
	ofs << "learning rate, " << learningRate << endl;
	ofs << confusion.format(CommaDel) << endl;
	ofs.close();
}

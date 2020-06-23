# Phänomen Detection Skript
[![CircleCI](https://circleci.com/gh/achenbachsven/learningSkript.svg?style=svg&circle-token=d93592aa7fbaab49a61bcd46306a44c607dae65c)](https://circleci.com/gh/achenbachsven/learningSkript/)
[![Launch binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/achenbachsven/learningSkript.git/4f70db1?urlpath=lab)
[![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)]()

The aim of this python script is to provide a simple, user-friendly and time-efficient solution for solving classification problems with a Neural Network (NN) library integrated in an Arm Cortex M4 Microcontroller (Infenion XMC 4700). Additionally the user has the ability to deploy the learned Keras NN weights by using an automatically generated interface header file written in the programming language C. This headerfile can be downloaded and integrated directly in a MCU Project environment by replacing a NN_weights.h file. The following Figure shows the schematic structure and workflow of the script.

![Screenshot](LearningSkriptWorkflow.png)

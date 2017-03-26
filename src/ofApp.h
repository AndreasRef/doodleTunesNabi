#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"
#include "ofxGui.h"
#include "ofxGrt.h"
#include "ofxCcv.h"
#include "ofxOsc.h"

#define HOST "localhost"
#define PORT 9000

using namespace ofxCv;
using namespace cv;


const vector<string> classNames = {"drums", "bass guitar", "saxophone", "keyboard"};


struct FoundSquare {
    ofImage img;
    int label = -1;
    cv::Rect rect;
    float area;
    void draw();
};

class ofApp : public ofBaseApp
{
public:
    void setup();
    void update();
    void draw();
    void exit();

    void setTrainingLabel(int & label_);
    void addSamplesToTrainingSet();
    void gatherFoundSquares();
    void trainClassifier();
    void classifyCurrentSamples();
    
    void addSamplesToTrainingSetNext();
    void classifyNext();

    void save();
    void load();

    int width, height;
    
    ofVideoGrabber cam;
    ContourFinder contourFinder, contourFinder2;
    ofFbo fbo;
    ofxCvGrayscaleImage grayImage;
    ofxCvColorImage colorImage;
    
    ofxOscSender sender;
    
    ofxPanel gui;
    ofParameter<float> minArea, maxArea, threshold;
    ofParameter<bool> holes;
    ofParameter<float> minArea2, maxArea2, threshold2;
    ofParameter<bool> holes2;
    ofParameter<int> nDilate;
    ofParameter<int> trainingLabel;
    ofxButton bAdd;
    ofxButton bTrain;
    ofxButton bClassify;
    ofxButton bSave, bLoad;
    ofxToggle bRunning;

    vector<FoundSquare> foundSquares;
    
    ClassificationData trainingData;
    GestureRecognitionPipeline pipeline;
    ofxCcv ccv;
    bool isTrained;
    bool toAddSamples;
    bool toClassify;

};

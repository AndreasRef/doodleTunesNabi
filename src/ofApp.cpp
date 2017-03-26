#include "ofApp.h"

//--------------------------------------------------------------
void FoundSquare::draw() {
    img.draw(0, 0);
    string labelStr = "no class";
    if (label > -1) {
        labelStr = "prediction: "+classNames[label];
    }
    ofDrawBitmapStringHighlight(labelStr, 0, img.getHeight()+20);
    ofDrawBitmapStringHighlight("{"+ofToString(rect.x)+", "+ofToString(rect.y)+", "+ofToString(rect.width)+", "+ofToString(rect.height)+"}", 0, img.getHeight()+40);
    ofDrawBitmapStringHighlight("area = "+ofToString(area), 0, img.getHeight()+60);
}

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetWindowShape(1600, 900);
    
    width = 640;
    height = 480;
    
    cam.setDeviceID(0);
    cam.setup(width, height);
    ccv.setup("image-net-2012.sqlite3");
    
    bAdd.addListener(this, &ofApp::addSamplesToTrainingSetNext);
    bTrain.addListener(this, &ofApp::trainClassifier);
    bClassify.addListener(this, &ofApp::classifyNext);
    bSave.addListener(this, &ofApp::save);
    bLoad.addListener(this, &ofApp::load);
    trainingLabel.addListener(this, &ofApp::setTrainingLabel);
    
    sender.setup(HOST, PORT);
    
    gui.setup();
    gui.setName("CV");
    ofParameterGroup gCvInitial, gCvMerge;
    gCvInitial.setName("CV initial");
    gCvInitial.add(minArea.set("Min area", 10, 1, 100));
    gCvInitial.add(maxArea.set("Max area", 200, 1, 500));
    gCvInitial.add(threshold.set("Threshold", 128, 0, 255));
    gCvInitial.add(holes.set("Holes", false));
    gCvMerge.setName("CV merge");
    gCvMerge.add(minArea2.set("Min area", 10, 1, 100));
    gCvMerge.add(maxArea2.set("Max area", 200, 1, 500));
    gCvMerge.add(threshold2.set("Threshold", 128, 0, 255));
    gCvMerge.add(holes2.set("Holes", false));
    gCvMerge.add(nDilate.set("Dilations", 1, 0, 8));
    gui.add(trainingLabel.set("Training Label", 0, 0, classNames.size()-1));
    gui.add(bAdd.setup("Add samples"));
    gui.add(bTrain.setup("Train"));
    gui.add(bRunning.setup("Run", false));
    gui.add(bClassify.setup("Classify"));
    gui.add(bSave.setup("Save"));
    gui.add(bLoad.setup("Load"));
    gui.add(gCvInitial);
    gui.add(gCvMerge);
    gui.setPosition(0, 0.75*height);
    gui.loadFromFile("settings.xml");
    
    fbo.allocate(width, height);
    colorImage.allocate(width, height);
    grayImage.allocate(width, height);
    isTrained = false;
    toAddSamples = false;
    toClassify = false;
    
    trainingData.setNumDimensions(4096);
    AdaBoost adaboost;
    adaboost.enableNullRejection(false);
    adaboost.setNullRejectionCoeff(3);
    pipeline.setClassifier(adaboost);
    
    
    //load on setup
    load();
}

//--------------------------------------------------------------
void ofApp::update(){
    cam.update();
    if(cam.isFrameNew())
    {
        // get grayscale image and threshold
        colorImage.setFromPixels(cam.getPixels());
        grayImage.setFromColorImage(colorImage);
        for (int i=0; i<nDilate; i++) {
            grayImage.erode_3x3();
        }
        grayImage.threshold(threshold);
        //grayImage.invert();
        
        // find initial contours
        contourFinder.setMinAreaRadius(minArea);
        contourFinder.setMaxAreaRadius(maxArea);
        contourFinder.setThreshold(127);
        contourFinder.findContours(grayImage);
        contourFinder.setFindHoles(holes);
        
        // draw all contour bounding boxes to FBO
        fbo.begin();
        ofClear(0, 255);
        ofFill();
        ofSetColor(255);
        for (int i=0; i<contourFinder.size(); i++) {
            //cv::Rect rect = contourFinder.getBoundingRect(i);
            //ofDrawRectangle(rect.x, rect.y, rect.width, rect.height);
            ofBeginShape();
            for (auto p : contourFinder.getContour(i)) {
                ofVertex(p.x, p.y);
            }
            ofEndShape();
        }
        fbo.end();
        ofPixels pixels;
        fbo.readToPixels(pixels);
        
        // find merged contours
        contourFinder2.setMinAreaRadius(minArea2);
        contourFinder2.setMaxAreaRadius(maxArea2);
        contourFinder2.setThreshold(threshold2);
        contourFinder2.findContours(pixels);
        contourFinder2.setFindHoles(holes2);
        
        if (toAddSamples) {
            addSamplesToTrainingSet();
            toAddSamples = false;
        }
        else if (isTrained && (bRunning || toClassify)) {
            classifyCurrentSamples();
            toClassify = false;
        }
    }
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(70);
    gui.draw();
    
    ofPushMatrix();
    ofScale(0.75, 0.75);
    
    // original
    ofPushMatrix();
    ofPushStyle();
    ofTranslate(0, 20);
    cam.draw(0, 0);
    ofDrawBitmapStringHighlight("original", 0, 0);
    ofPopMatrix();
    ofPopStyle();
    
    // thresholded
    ofPushMatrix();
    ofPushStyle();
    ofTranslate(width, 20);
    grayImage.draw(0, 0);
    ofSetColor(0, 255, 0);
    contourFinder.draw();
    ofDrawBitmapStringHighlight("thresholded", 0, 0);
    ofPopMatrix();
    ofPopStyle();
    
    // merged
    ofPushMatrix();
    ofPushStyle();
    ofTranslate(2*width, 20);
    fbo.draw(0, 0);
    ofSetColor(0, 255, 0);
    contourFinder2.draw();
    ofDrawBitmapStringHighlight("merged", 0, 0);
    ofPopMatrix();
    ofPopStyle();
    
    ofPopMatrix();
    
    // draw tiles
    ofPushMatrix();
    ofPushStyle();
    ofTranslate(210, 0.75*height+60);
    for (int i=0; i<foundSquares.size(); i++) {
        ofPushMatrix();
        ofTranslate(226*i, 0);
        foundSquares[i].draw();
        ofPopMatrix();
    }
    ofPopMatrix();
    ofPopStyle();
    
    
}

//--------------------------------------------------------------
void ofApp::exit() {
    gui.saveToFile("settings.xml");
}

//--------------------------------------------------------------
void ofApp::gatherFoundSquares() {
    foundSquares.clear();
    for (int i=0; i<contourFinder2.size(); i++) {
        FoundSquare fs;
        fs.rect = contourFinder2.getBoundingRect(i);
        fs.area = contourFinder2.getContourArea(i);
        fs.img.setFromPixels(cam.getPixels());
        fs.img.crop(fs.rect.x, fs.rect.y, fs.rect.width, fs.rect.height);
        fs.img.resize(224, 224);
        foundSquares.push_back(fs);
    }
}

//--------------------------------------------------------------
void ofApp::addSamplesToTrainingSet() {
    ofLog(OF_LOG_NOTICE, "Adding samples...");
    gatherFoundSquares();
    for (int i=0; i<foundSquares.size(); i++) {
        vector<float> encoding = ccv.encode(foundSquares[i].img, ccv.numLayers()-1);
        VectorFloat inputVector(encoding.size());
        for (int i=0; i<encoding.size(); i++) inputVector[i] = encoding[i];
        trainingData.addSample(trainingLabel, inputVector);
        ofLog(OF_LOG_NOTICE, " Added sample #"+ofToString(i)+" label="+ofToString(trainingLabel));
    }
}

//--------------------------------------------------------------
void ofApp::trainClassifier() {
    ofLog(OF_LOG_NOTICE, "Training...");
    if (pipeline.train(trainingData)){
        cout << "getNumClasses: " << pipeline.getNumClasses() << endl;
    }
    isTrained = true;
    ofLog(OF_LOG_NOTICE, "Done training...");
}


//--------------------------------------------------------------
void ofApp::classifyCurrentSamples() {
    int nInstruments = 4;
    vector<int>instrumentCount;
    vector<float>instrumentArea;
    
    for (int i = 0; i< nInstruments; i++) {
        instrumentCount.push_back(0);
        instrumentArea.push_back(0);
    }
    
    ofLog(OF_LOG_NOTICE, "Classifiying "+ofToString(ofGetFrameNum()));
    gatherFoundSquares();
    float maxArea = 0.0;
    for (int i=0; i<foundSquares.size(); i++) {
        vector<float> encoding = ccv.encode(foundSquares[i].img, ccv.numLayers()-1);
        VectorFloat inputVector(encoding.size());
        for (int i=0; i<encoding.size(); i++) inputVector[i] = encoding[i];
        if (pipeline.predict(inputVector)) {
            int label = pipeline.getPredictedClassLabel();
            foundSquares[i].label = label;
            instrumentCount[label]++;
            instrumentArea[label] = max(instrumentArea[label], foundSquares[i].area);
            maxArea = max(maxArea, foundSquares[i].area);
        }
    }
    
    
    //Send OSC messages to Ableton via liveOSC commands
    for (int i = 0; i<nInstruments; i++) {
        if (instrumentCount[i] > 0) {
            
            //Launch the clips
            ofxOscMessage m;
            m.setAddress("/live/play/clip");
            m.addIntArg(i); //Set the track
            m.addIntArg(instrumentCount[i]-1); //Set the clip
            sender.sendMessage(m, false);
            
            //Set the track volume based on size of the instruments
            float trackVol = 0.75*instrumentArea[i]/maxArea; 
            ofxOscMessage m2;
            m2.setAddress("/live/volume");
            m2.addIntArg(i);
            m2.addFloatArg(ofMap(trackVol,0,0.75,0.55,0.75));
            sender.sendMessage(m2, false);
            
            
        }
        else {
            ofxOscMessage m;
            m.setAddress("/live/stop/track");
            m.addIntArg(i); //Set the track
            sender.sendMessage(m, false);
        }
    }
}

//--------------------------------------------------------------
void ofApp::setTrainingLabel(int & label_) {
    trainingLabel.setName(classNames[label_]);
}

//--------------------------------------------------------------
void ofApp::save() {
    trainingData.save(ofToDataPath("TrainingData.grt"));
}

//--------------------------------------------------------------
void ofApp::load() {
    trainingData.load(ofToDataPath("TrainingData.grt"));
    trainClassifier();
}

//--------------------------------------------------------------
void ofApp::classifyNext() {
    toClassify = true;
}

//--------------------------------------------------------------
void ofApp::addSamplesToTrainingSetNext() {
    toAddSamples = true;
}

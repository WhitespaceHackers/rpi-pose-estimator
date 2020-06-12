const posenet = require('@tensorflow-models/posenet');
const tfnode  = require('@tensorflow/tfjs-node');
const fs = require('fs')

const height = 200;
const width = 257;

const readImage = path => {
    const imageBuffer = fs.readFileSync(path);

    const tfimage = tfnode.node.decodeImage(imageBuffer); //default #channel 4
    return tfimage;
}

async function estimatePoseOnImage(imageElement) {
    // load the posenet model from a checkpoint
    // const net = await posenet.load();
    const net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: 257, height: 200 },
        quantBytes: 2
    });

    const pose = await net.estimateSinglePose(imageElement, {
        flipHorizontal: false
    });
    return pose;
}

const RaspiCam = require("raspicam");

const camera = new RaspiCam({
    mode: "photo",
    output: "/home/pi/hackathon/whitespace/images/photos_%d.jpg",
    timeout: 100000,
    timelapse: 10000,
    width: 640,
    height: 480,
    nopreview: true,

});

//to take a snapshot, start a timelapse or video recording
camera.start( );

//listen for the "start" event triggered when the start method has been successfully initiated
camera.on("start", function(){
    //do stuff
});

//listen for the "read" event triggered when each new photo/video is saved
camera.on("read", function(err, timestamp, filename){
    //do stuff
    console.log("read callback");
    console.log(filename);

    const pose = estimatePoseOnImage(readImage(`images/${filename}`));

    pose.then(data => {
        console.log('score :' + data.score);
        for (point in data.keypoints) {
            const _point = data.keypoints[point];
            console.log('body part: ' + _point.part);
            console.log('score: ' + _point.score);
            console.log('coordinates: (' + _point.position.x + ', ' + _point.position.y + ')');
            console.log();
        };
        var json = JSON.stringify(data);
        fs.writeFile(`pose_estimates/${filename.replace('jpg','json')}`, json, 'utf8', () => {});
    })
        .catch((err) => {
            console.log('error');
            console.log(err);
        });

});

//listen for the "stop" event triggered when the stop method was called
camera.on("stop", function(){
    //do stuff
});

//listen for the process to exit when the timeout has been reached
camera.on("exit", function(){
    //do stuff
});

//to stop a timelapse or video recording
// setTimeout(() => { camera.stop(); }, 20000);

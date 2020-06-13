const posenet = require('@tensorflow-models/posenet');
const tfnode  = require('@tensorflow/tfjs-node');
const fs = require('fs');
const axios = require('axios');
const regression = require('regression');

const readImage = path => {
    const imageBuffer = fs.readFileSync(path);

    const tfimage = tfnode.node.decodeImage(imageBuffer); //default #channel 4
    return tfimage;
}

async function estimatePoseOnImage(imageElement) {
    // load the posenet model from a checkpoint
    const net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: 200, height: 200 },
        quantBytes: 1
    });

    const pose = await net.estimateSinglePose(imageElement, {
        flipHorizontal: false
    });
    return pose;
}

function thresholdFallDectection(pose) {
    const threshold = 0.1;
    const keypoints = pose.keypoints;
    let y_coordinates = 0;
    let counter = 0;
    for (p in keypoints) {
        if (keypoints[p].score > threshold) {
            y_coordinates += keypoints[p]['position']['y'];
            counter += 1;
        }
    }
    const avg_y = y_coordinates / counter;
    return avg_y > 350;
}

function linRegFallDectection(pose) {
    const threshold = 0.05;
    const keypoints = pose.keypoints;
    let coordinates = [];
    for (p in keypoints) {
        if (keypoints[p].score > threshold) {
            const x = keypoints[p]['position']['x'];
            const y = keypoints[p]['position']['y'];
            coordinates.push([x,y]);
        }
    }
    const linreg = regression.linear(coordinates);
    const grad = Math.abs(linreg.equation[0]);
    const r2 = linreg.r2;
    return grad < 0.5;
}

const RaspiCam = require("raspicam");

const camera = new RaspiCam({
    mode: "photo",
    output: "/home/pi/hackathon/whitespace/images/photos.jpg",
    timeout: 1000 * 60 * 24 * 7,
    timelapse: 4000,
    width: 640,
    height: 480,
    nopreview: true,

});

//to take a snapshot, start a timelapse or video recording
camera.start( );

//listen for the "start" event triggered when the start method has been successfully initiated
camera.on("start", function() {
    //do stuff
});

//listen for the "read" event triggered when each new photo/video is saved
camera.on("read", function(err, timestamp, filename) {
    //do stuff
    const start = Date.now();
    console.log(filename);

    const pose = estimatePoseOnImage(readImage(`images/${filename}`));

    pose.then(data => {
        // console.log(data);
        const fall = thresholdFallDectection(data);
        console.log('fall detected: : ' + fall);
        data.fall = fall;
        const json = JSON.stringify(data);
        fs.writeFile(`pose_estimates/${filename.replace('jpg','json')}`, json, 'utf8', () => {});

        // send data to server
        axios.post('https://mdjj-api.us-south.cf.appdomain.cloud/rpi', data)
            .then(res => {
                console.log(filename + ' sent successfully');
                // console.log(res);
            })
            .catch(err => {
                console.log('unsuccessfully');
                console.log(err);
            });

        // log execution time
        const end = Date.now();
        console.log('execution time: ' + (end - start) + 'ms');
    })
        .catch((err) => {
            console.log('error');
            console.log(err);
        });


});

//listen for the "stop" event triggered when the stop method was called
camera.on("stop", function() {
    //do stuff
});

//listen for the process to exit when the timeout has been reached
camera.on("exit", function() {
    //do stuff
});

//to stop a timelapse or video recording
// setTimeout(() => { camera.stop(); }, 20000);

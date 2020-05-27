'use strict';
const express = require('express')
const path = require('path')

const no = require('@tensorflow/tfjs-node');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const fetch = require('node-fetch')
const log4js = require('log4js');

const logger = log4js.getLogger(`RECO`);
logger.level = 'info';

const PORT = process.env.PORT || 5000;

const Canvas = canvas.Canvas, Image = canvas.Image, ImageData = canvas.ImageData;

faceapi.env.monkeyPatch({Canvas: Canvas, Image: Image, ImageData: ImageData});
faceapi.env.monkeyPatch({fetch: fetch});

// const faceDetectionNet = faceapi.nets.ssdMobilenetv1;
const faceDetectionNet = faceapi.nets.tinyFaceDetector

// SsdMobilenetv1Options
const minConfidence = 0.5;
// TinyFaceDetectorOptions
const inputSize = 416;
let loaded = false

const scoreThreshold = 0.5;

function getFaceDetectorOptions(net) {
    return net === faceapi.nets.ssdMobilenetv1
        ? new faceapi.SsdMobilenetv1Options({minConfidence: minConfidence})
        : new faceapi.TinyFaceDetectorOptions({inputSize: inputSize, scoreThreshold: scoreThreshold});

}

const faceDetectionOptions = getFaceDetectorOptions(faceDetectionNet);

async function loadModels() {
    if (loaded) return;

    loaded = true
    logger.info('LOAD MODELS...')

    // await faceDetectionNet.loadFromDisk('./weights');
    // await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights');
    // await faceapi.nets.ageGenderNet.loadFromDisk('./weights');
    // await faceDetectionNet.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/ssd_mobilenetv1_model-weights_manifest.json');
    await faceDetectionNet.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-weights_manifest.json');
    await faceapi.nets.faceLandmark68Net.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_model-weights_manifest.json');

    await faceapi.nets.ageGenderNet.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/age_gender_model-weights_manifest.json');

    await faceapi.nets.faceRecognitionNet.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_recognition_model-weights_manifest.json')

}

async function getByUrl(url, reco = '0') {
    let img;

    try {
        img = await canvas.loadImage(url)
    } catch (e) {
        logger.error(`can not get by URL ${e.message}: ${url}`)
        return
    }

    const data = []
    if (reco === '1') {
        const results = await faceapi.detectAllFaces(img, faceDetectionOptions)
            .withFaceLandmarks()
            .withFaceDescriptors()
        results.forEach(result => {
            const {age, gender, genderProbability} = result
            data.push(result.descriptor)
        })
    } else if (reco === '2') {

        const results = await faceapi.detectAllFaces(img, faceDetectionOptions)
            .withFaceLandmarks()
            .withAgeAndGender()
            .withFaceDescriptors()
        results.forEach(result => {
            data.push(result)
        })

    } else {
        const results = await faceapi.detectAllFaces(img, faceDetectionOptions)
            .withFaceLandmarks()
            .withAgeAndGender()
        results.forEach(result => {
            const {age, gender, genderProbability} = result
            data.push({
                age: age,
                gender: gender,
                genderProbability: genderProbability
            })
        })

    }
    return data;

}

express()
    .get('/', async (req, res) => {

        if (!req.query.url) {

            res.status(400).end('no url')
            logger.error('no url')

        } else {

            if (!loaded) await loadModels();

            let data = await getByUrl(req.query.url, req.query.reco)

            if (data === undefined) {
                logger.warn(`no data by ${req.query.url}`)
                res.status(404).end(`no data by ${req.query.url}`)
            } else {
                logger.debug(data)
                res.send(JSON.stringify(data)).end();
            }

        }
    })
    .post('/', async (req, res) => {
        throw 'RESTART';
    })
    .listen(PORT, () => console.log(`Listening on ${PORT}`))

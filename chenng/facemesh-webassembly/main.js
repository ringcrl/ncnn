var Module = {};

var has_simd;
var has_threads;

var wasmModuleLoaded = false;
var wasmModuleLoadedCallbacks = [];

Module.onRuntimeInitialized = function () {
  wasmModuleLoaded = true;
  for (var i = 0; i < wasmModuleLoadedCallbacks.length; i++) {
    wasmModuleLoadedCallbacks[i]();
  }
};

wasmFeatureDetect.simd().then((simdSupported) => {
  has_simd = simdSupported;

  wasmFeatureDetect.threads().then((threadsSupported) => {
    has_threads = threadsSupported;

    if (has_simd) {
      if (has_threads) {
        nanodet_module_name = "facemesh-simd-threads";
      } else {
        nanodet_module_name = "facemesh-simd";
      }
    } else {
      if (has_threads) {
        nanodet_module_name = "facemesh-threads";
      } else {
        nanodet_module_name = "facemesh-basic";
      }
    }

    console.log("load " + nanodet_module_name);

    var nanodetwasm = nanodet_module_name + ".wasm";
    var nanodetjs = nanodet_module_name + ".js";

    fetch(nanodetwasm)
      .then((response) => response.arrayBuffer())
      .then((buffer) => {
        Module.wasmBinary = buffer;
        var script = document.createElement("script");
        script.src = nanodetjs;
        script.onload = function () {
          console.log("Emscripten boilerplate loaded.");
        };
        document.body.appendChild(script);
      });
  });
});

var shouldFaceUser = true;
var stream = null;
var w = 640;
var h = 480;

var dst = null;
var resultarray = null;
var resultbuffer = null;
window.addEventListener("DOMContentLoaded", function () {
  var isStreaming = false;
  switchcamerabtn = document.getElementById("switch-camera-btn");
  video = document.getElementById("video");
  canvas = document.getElementById("canvas");
  ctx = canvas.getContext("2d");

  // Wait until the video stream canvas play
  video.addEventListener(
    "canplay",
    function (e) {
      if (!isStreaming) {
        // videoWidth isn't always set correctly in all browsers
        if (video.videoWidth > 0)
          h = video.videoHeight / (video.videoWidth / w);
        canvas.setAttribute("width", w);
        canvas.setAttribute("height", h);
        isStreaming = true;
      }
    },
    false
  );

  // Wait for the video to start to play
  video.addEventListener("play", function () {
    //Setup image memory
    var id = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var d = id.data;

    if (wasmModuleLoaded) {
      mallocAndCallSFilter();
    } else {
      wasmModuleLoadedCallbacks.push(mallocAndCallSFilter);
    }

    function mallocAndCallSFilter() {
      if (dst != null) {
        _free(dst);
        dst = null;
      }

      dst = _malloc(d.length);

      //console.log("What " + d.length);

      sFilter();
    }
  });

  // check whether we can use facingMode
  var supports = navigator.mediaDevices.getSupportedConstraints();
  if (supports["facingMode"] === true) {
    switchcamerabtn.disabled = false;
  }

  switchcamerabtn.addEventListener("click", function () {
    if (stream == null) return;

    stream.getTracks().forEach((t) => {
      t.stop();
    });

    shouldFaceUser = !shouldFaceUser;
    capture();
  });

  capture();
});

function capture() {
  var constraints = {
    audio: false,
    video: {
      width: 640,
      height: 480,
      facingMode: shouldFaceUser ? "user" : "environment",
    },
  };
  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(function (mediaStream) {
      var video = document.querySelector("video");
      stream = mediaStream;
      video.srcObject = mediaStream;
      video.onloadedmetadata = function (e) {
        video.play();
      };
    })
    .catch(function (err) {
      console.log(err.message);
    });
}

function ncnn_nanodet() {
  var canvas = document.getElementById("canvas");
  var ctx = canvas.getContext("2d");

  var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  var data = imageData.data;

  HEAPU8.set(data, dst);

  _nanodet_ncnn(dst, canvas.width, canvas.height);

  var result = HEAPU8.subarray(dst, dst + data.length);
  imageData.data.set(result);
  ctx.putImageData(imageData, 0, 0);
}

//Request Animation Frame function
var sFilter = function () {
  if (video.paused || video.ended) return;

  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(video, 0, 0, w, h);

  ncnn_nanodet();

  window.requestAnimationFrame(sFilter);
};
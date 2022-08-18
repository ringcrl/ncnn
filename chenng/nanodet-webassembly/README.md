# nanodet-webassembly


## 编译

```sh
# emscripten
# 只能是这个版本，低了 404，高了 error: undefined symbol
# https://github.com/emscripten-core/emscripten/issues/16826
~/Documents/saga/emsdk/emsdk install 2.0.21
~/Documents/saga/emsdk/emsdk activate 2.0.21
source ~/Documents/saga/emsdk/emsdk_env.sh

# 更新 ncnn-webassembly：https://github.com/Tencent/ncnn/releases

# 编译四个版本的 webassembly
./build.sh

```

## 运行

```sh
npm i

npm run dev
```

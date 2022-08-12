set -e

mkdir -p build
cd build

cmake -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DVulkan_INCLUDE_DIR=`pwd`/../vulkansdk-macos-1.2.189.0/MoltenVK/include \
    -DVulkan_LIBRARY=`pwd`/../vulkansdk-macos-1.2.189.0/MoltenVK/dylib/macOS/libMoltenVK.dylib \
    -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..

cmake --build . -j 16
cmake --build . --target install

rm -rf ../.test/face_landmark 
cp -r ./examples/face_landmark ../.test/face_landmark 

cd ../.test/
./face_landmark

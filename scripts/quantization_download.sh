base=$(dirname "$0")
cur=$(pwd)
cd ${base}
mkdir -p ../quantization
cd ../quantization
# curl http://www.quantize.maths-fi.com/sites/default/files/quantization_grids/mult_dimensional_grids.zip --output mult_dimensional_grids.zip
cp ~/Downloads/mult_dimensional_grids.zip ./
unzip -o mult_dimensional_grids.zip
rm mult_dimensional_grids.zip
cd ${cur}

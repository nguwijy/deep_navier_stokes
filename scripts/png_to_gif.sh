# make sure to cd to the right log directory, then run bash ../../scripts/png_to_gif.sh
convert -delay 10 $(for i in $(seq -f "%04g" 0 10 2999); do echo epoch_${i}.png; done) -loop 0 animation.gif

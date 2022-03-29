# make sure to cd to the right log directory, then run bash ../../scripts/png_to_gif.sh
convert -delay 10 -loop 0 plot/*png animation.gif

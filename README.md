# radiative_transfer_amcvn

While in Professor Blaes's Lab, I was in charge of reproducing the line the
line spectrum of AM CVn systems. These systems are merely a binary system with
a accretion disk. The important feature about these systems is that they are 
small compared to other accretion disks out in space, so we are able to numeri-
cally simulate the whole disk. 

I wrote a code that could solve for the frequency specturm of the simulated 
accretion disk at a given time using 3-dimensional data. 

## Running the code

There are many factors to change depending what you want to run. If you would
like to change the data of the simulation, then place your data in the data 
directory and change the variable `txt_file` in the main code `amcvn_3d.py`.

You can also choose the viewing angle, number of photon paths and the amount of
points per line.

## Outfiles

The script will return three graphs, one for the photon paths illustrated in a 
3D plot, the frequency spectrum over the whole frequency range, and the 
frequency spectrum zoomed in where the absorption lines would be observed if
any. 

To see some examples, go to the plots directory and look at how the frequency
spectrum changes so much just by changing the viewing angle.

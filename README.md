# KEH_STT

## Description

Sample code for obtaining static Neutron Star solutions in the Damour-Esposito-Farese (DEF) scalar-tensor model using the KEH/CST numerical scheme.

## Usage

run python KEH_STT.py

KEH_STT.py  7 inputs in-turn:

1. The EoS file name (e.g. eosSLY.txt).
2. The central energy density in CGS/10^15 (e.g 1.2).
3. The STT coupling constant (e.g. -5.0).
4. The relative error for iteration scheme (e.g 1e-05)
5. The number of maximum iteraions (e.g 200)
6. The relaxation factor (e.g 0.2)
7. The print option 0 or 1:
    -  0: Prints gravitational mass M and radius R.
    -  1: Prints (0) along with the distance, metric, scalar, energy density and pressure profiles.

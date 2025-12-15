#!/bin/bash

# Script to download GLADE+ catalog and extract specific columns
# Downloads from ELTE server and extracts columns 6-12 (RA, Dec, distance, redshift, etc.)

echo "Downloading GLADE+ catalog data..."
echo "We are only downloading a subset of the whole dataset (columns 6-12: RA, Dec, distance, redshift, etc)."

# Download and process the GLADE+ catalog
wget -qO- http://elysium.elte.hu/~dalyag/GLADE+.txt | \
awk '{print $6, $7, $8, $9, $10, $11}' OFS='\t' > GLADE_plus_subset.txt

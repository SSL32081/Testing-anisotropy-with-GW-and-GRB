all: data figures

figures: data figures_2mass
figures_2mass: figures/2mass_galaxy_catalog_spec_ra_dec.pdf figures/2mass_galaxy_catalog_spec_l_b.pdf
figures/2mass_galaxy_catalog_spec_ra_dec.pdf figures/2mass_galaxy_catalog_spec_l_b.pdf: 2mass/plot_samples.py
	mkdir -p figures
	cd 2mass && \
	uv run plot_samples.py && \
	mv 2mass_galaxy_catalog_spec_ra_dec.pdf ../figures/2mass_galaxy_catalog_spec_ra_dec.pdf && \
	mv 2mass_galaxy_catalog_spec_l_b.pdf ../figures/2mass_galaxy_catalog_spec_l_b.pdf


data: 2mass
2mass: 
	uv run gra data get 2mass spectroscopic

clean:
	rm -rf data/2mass
	rm -rf figures/2mass_galaxy_catalog_spec_ra_dec.pdf
	rm -rf figures/2mass_galaxy_catalog_spec_l_b.pdf


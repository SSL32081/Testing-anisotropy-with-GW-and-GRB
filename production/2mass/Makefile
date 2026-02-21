all: data figures

figures: data figures_2mass
figures_2mass: figures/2mass_galaxy_catalog_spec_ra_dec.pdf figures/2mass_galaxy_catalog_spec_l_b.pdf \
	figures/example_synthetic_2mass_data.pdf figures/actual_2mass_data.pdf figures/2mass_angular_power_spectrum_comparison.pdf figures/2mass_angular_power_spectrum_comparison_all_synthetic.pdf
figures/2mass_galaxy_catalog_spec_ra_dec.pdf figures/2mass_galaxy_catalog_spec_l_b.pdf: 2mass/plot_samples.py
	mkdir -p figures
	cd 2mass && \
	uv run plot_samples.py && \
	mv 2mass_galaxy_catalog_spec_ra_dec.pdf ../figures/2mass_galaxy_catalog_spec_ra_dec.pdf && \
	mv 2mass_galaxy_catalog_spec_l_b.pdf ../figures/2mass_galaxy_catalog_spec_l_b.pdf
figures/example_synthetic_2mass_data.pdf figures/actual_2mass_data.pdf figures/2mass_angular_power_spectrum_comparison.pdf figures/2mass_angular_power_spectrum_comparison_all_synthetic.pdf 2mass/synthetic_2mass_catalog.npz 2mass/synthetic_2mass_spectra.npz: 2mass/generate_synthetic_catalog.py
	mkdir -p figures
	cd 2mass && \
	uv run generate_synthetic_catalog.py && \
	mv example_synthetic_2mass_data.pdf ../figures/ && \
	mv actual_2mass_data.pdf ../figures/ && \
	mv 2mass_angular_power_spectrum_comparison.pdf ../figures/ && \
	mv 2mass_angular_power_spectrum_comparison_all_synthetic.pdf ../figures/

data: 2mass
2mass: 2mass/2mass_galaxy_catalog_spec.csv
2mass/2mass_galaxy_catalog_spec.csv:
	uv run gra data get 2mass spectroscopic

clean:
	rm -f 2mass/2mass_galaxy_catalog_spec.csv
	rm -f 2mass/*.npz
	rm -rf figures/2mass_galaxy_catalog_spec_ra_dec.pdf
	rm -rf figures/2mass_galaxy_catalog_spec_l_b.pdf


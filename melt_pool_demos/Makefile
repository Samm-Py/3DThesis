PYTHON ?= python3
ROOT := $(abspath ..)
THESIS_BIN ?= $(ROOT)/build/bin/3DThesis
THESIS_BIN_OTI ?= $(ROOT)/build-oti/bin/3DThesis
export THESIS_BIN THESIS_BIN_OTI

.PHONY: help build build-oti validate raster single_track two_tracks square triangle figures paper notes

help:
	@echo "Publication workflows:"
	@echo "  make build        plain-double solver"
	@echo "  make build-oti    OTI sensitivity solver"
	@echo "  make validate     verify the committed result tables against the paper"
	@echo "  make raster       reproduce the 1 mm raster study"
	@echo "  make single_track single-track base case and calibration"
	@echo "  make two_tracks   two-track continuous P/sigma/v case and block-length choice"
	@echo "  make square       101-line continuous square: optimize, replay, figures, paper"
	@echo "  make triangle     87-line continuous triangle: optimize, replay, figures, paper"
	@echo "  make figures      regenerate plots from existing results"
	@echo "  make paper        compile the working paper (raster/doc/paper/main.tex)"
	@echo "  make notes        compile the technical notes (raster/doc/optimization_notes.tex)"

build:
	cmake -S $(ROOT) -B $(ROOT)/build -DCMAKE_BUILD_TYPE=Release -DTHESIS_ENABLE_MPI=OFF
	cmake --build $(ROOT)/build -j

build-oti:
	cmake -S $(ROOT) -B $(ROOT)/build-oti -DCMAKE_BUILD_TYPE=Release -DTHESIS_ENABLE_MPI=OFF -DTHESIS_ENABLE_OTI=ON
	cmake --build $(ROOT)/build-oti -j

validate:
	$(PYTHON) validate_results.py

raster:
	cd raster && $(PYTHON) find_min_dwell.py
	cd raster && $(PYTHON) run_baseline.py
	cd raster && $(PYTHON) run_optimized.py
	cd raster && $(PYTHON) make_plots.py

single_track:
	cd single_track && $(PYTHON) make_case.py --out cases/case_base --name Track
	cd single_track/cases/case_base && $(THESIS_BIN) ./ParamInput.txt
	cd single_track && $(PYTHON) ../common/calibrate.py cases/case_base Track 5e-5
	cd single_track && $(PYTHON) ../common/fullfield_stats.py cases/case_base Track
	cd single_track && $(PYTHON) ../common/make_plots.py fig17 case_base Track
	cd single_track && $(PYTHON) ../common/make_plots.py fig16 case_base Track
	cd single_track && $(PYTHON) ../common/run_baseline.py zero
	cd single_track && $(PYTHON) ../common/run_optimized.py zero
	cd single_track && $(PYTHON) verify_optimized.py

two_tracks:
	cd two_tracks && PYTHON=$(PYTHON) ./run_study.sh

# The square and triangle paper records: the greedy continuous P/sigma/v
# controller on 1 mm blocks, the 50/10/1 um tracked replays of the nominal
# baseline and the optimized schedule, the figure set, the paper-figure copy,
# and the LaTeX build. The single-track calibration record
# (results/calibration/, target_zero.json) is committed; regenerate it first
# with the make_case.py --single-track + calibrate.py workflow in the study
# README if starting from a clean checkout.
square:
	cd square && $(PYTHON) verify_baseline.py --policy zero --resx 50e-6 --resy 10e-6 --resz 1e-6
	cd square && PYTHON_BIN=$(PYTHON) ./run_continuous_study.sh

triangle:
	cd triangle && $(PYTHON) verify_baseline.py --policy zero --resx 50e-6 --resy 10e-6 --resz 1e-6
	cd triangle && PYTHON_BIN=$(PYTHON) ./run_continuous_study.sh

figures:
	cd raster && $(PYTHON) make_plots.py
	cd two_tracks && $(PYTHON) make_publication_figures.py
	cd square && $(PYTHON) make_greedy_figures.py --policy zero
	cd square && $(PYTHON) make_paper_maps.py --policy zero
	cd triangle && $(PYTHON) make_greedy_figures.py --policy zero
	cd triangle && $(PYTHON) make_paper_maps.py --policy zero

paper:
	cd raster/doc/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd raster/doc/paper && bibtex main || true
	cd raster/doc/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd raster/doc/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex

notes:
	cd raster/doc && pdflatex -interaction=nonstopmode -halt-on-error optimization_notes.tex
	cd raster/doc && pdflatex -interaction=nonstopmode -halt-on-error optimization_notes.tex

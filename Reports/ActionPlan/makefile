# Makefile for compiling LaTeX documents with xelatex and organizing output files

# Define the main document name (without extension)
EXAMPLES = report brief-action-plan

# List of LaTeX style files
STYLE = *.sty

# Define the directory for build files
BUILD_DIR = build

# Define the xelatex command and related
LATEX = latexmk
LATEX_FLAGS = -pdf -xelatex -interaction=nonstopmode -shell-escape -file-line-error --quiet
PRETTIER = prettier
PRETTIER_PLUGIN = /opt/homebrew/lib/node_modules/prettier-plugin-latex/dist/prettier-plugin-latex.js

# List of auxiliary files to remove
AUX_FILES = *.aux *.log *.out *.toc *.nav *.snm *.vrb *.fls *.fdb_latexmk

# Remove the default suffixes
.SUFFIXES:

# Default target to compile the documents
all: $(foreach example, $(EXAMPLES), $(BUILD_DIR)/$(example).pdf)

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	@echo "Created build directory"

# Compile the documents using xelatex in batch mode and place output in the build directory
$(BUILD_DIR)/%.pdf: %.tex $(STYLE) | $(BUILD_DIR)
	@echo "Compiling $<"
	@$(LATEX) $(LATEX_FLAGS) -output-directory=$(BUILD_DIR) $<
	@if [ $$? -eq 0 ]; then $(MAKE) clean-aux; fi
	@echo "Compilation complete - $(basename $<).pdf"

# Format the LaTeX source files using prettier
format:
	$(foreach example, $(EXAMPLES), $(PRETTIER) --write --plugin=$(PRETTIER_PLUGIN) $(example).tex;)

clean-aux:
	$(foreach aux, $(AUX_FILES), rm -f $(BUILD_DIR)/$(aux);)

clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all format clean clean-aux

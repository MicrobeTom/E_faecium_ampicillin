# ---------------------------------------------------------------------------
# The code below has been adapted from the **maldi_amr** project by the Borgwardt Lab
# (https://github.com/BorgwardtLab/maldi_amr) licensed under the
# BSD 3‑Clause "New" License.  Copyright (c) 2020, Caroline V. Weis, Bastian Rieck, Aline Cuénod
#
# This derivative work retains the BSD 3‑Clause license; redistribution of this
# file must include the above notice and the full license text.
# ---------------------------------------------------------------------------

library("MALDIquant")
library("MALDIquantForeign")
library(stringr)

options(warn=0)

#########################################
# Define paths and configurations
#########################################


SINK_FILE <- paste('', Sys.Date(), '', sep = '')
sink(SINK_FILE, append = FALSE, split = FALSE)

# Directories containing spectra
DIR1 <- "" # ampicillin-susceptible spectra folder
DIR2 <- "" # ampicillin-resistant spectra folder

# Directories to store final aligned spectra
ECFMAS_OUT_DIR <- "" # aligned ampicillin-susceptible spectra folder
ECFMAR_OUT_DIR <- "" # aligned ampicillin-resistant spectra folder

# Collect all directories for processing
directories <- list(DIR1, DIR2)

# List to store all preprocessed spectra
spectra_list <- list(DIR1 = list(), DIR2 = list())

#########################################
# Helper function to process a directory
#########################################

process_directory <- function(dir_path) {
  list_files <- list.files(path = dir_path, pattern = "\\.txt$", recursive = TRUE)
  num_files <- length(list_files)
  
  print(paste("Processing directory:", dir_path))
  print(paste("Number of files:", num_files))
  
  processed_spectra <- list()
  emptyspec <- c()
  idvec <- c()
  
  for (j in seq_len(num_files)) {
    thisfilepath <- file.path(dir_path, list_files[j])
    # Extract first directory as ID
    fileid <- paste(unlist(strsplit(list_files[j], split = "/"))[1], "fid", sep = "")
    print(fileid)
    
    myspec_df <- read.table(
      file       = thisfilepath,
      header     = FALSE,
      comment.char = "#",
      blank.lines.skip = TRUE
    )
    
    # Skip if spectra is empty
    if (nrow(myspec_df) == 0) {
      print("Spectra is empty")
      emptyspec <- c(emptyspec, fileid)
      next
    }
    
    # Skip if spectra already processed
    if (fileid %in% idvec) {
      print("Sample already processed - skipping duplicate")
      next
    } else {
      idvec <- c(idvec, fileid)
    }
    
    myspec <- createMassSpectrum(mass=myspec_df[,1],
                                 intensity=myspec_df[,2],
                                 metaData=list(file=thisfilepath))
    
    processed_spectra[[fileid]] <- myspec
  }
  
  return(processed_spectra)
}

#########################################
# Process each directory
#########################################

# Process DIR1 (red spectra)
spectra_list$DIR1 <- process_directory(DIR1)

# Process DIR2 (blue spectra)
spectra_list$DIR2 <- process_directory(DIR2)

# Combine all spectra into a single list
all_spectra <- c(spectra_list$DIR1, spectra_list$DIR2)

#########################################
# Detect Peaks and Calculate Reference Spectrum
#########################################

# For nested cross validation, align spectra within the respective dataset and create reference peak list
# For external validation, align spectra of the test dataset to the training dataset by choosing the ref_file from the training dataset's nested CV alignment
# For details, see README.md

ref_file <- ""
min_freq_peaks <- 0.90  # Peaks must appear in 90% of spectra
tolerance <- 0.002      # Matching tolerance for m/z values

if (file.exists(ref_file)) {
  # Load precomputed reference spectrum
  reference_peaks_df <- read.table(ref_file, header = TRUE)
  reference_peaks <- createMassPeaks(reference_peaks_df$mass, reference_peaks_df$intensity)
  print("Loaded precomputed reference spectrum.")
} else {

  # Detect peaks in all spectra
  print("Calculating reference spectrum.")
  peaks_list <- detectPeaks(all_spectra, method = "MAD", halfWindowSize = 20, SNR = 3)
  
  # Calculate reference peaks
  reference_peaks <- referencePeaks(peaks_list, method = "strict",
                                    minFrequency = min_freq_peaks,
                                    tolerance = tolerance)
  print("Reference spectrum calculated.")
  
  # Save reference spectrum to file
  data_matrix <- data.frame(mass = mass(reference_peaks), intensity = intensity(reference_peaks))
  write.table(data_matrix, ref_file, sep = " ", row.names = FALSE, col.names = c("mass", "intensity"))
  print("Reference spectrum saved.")
}

#########################################
# Align Spectra to Reference Spectrum
#########################################

# First, detect peaks for all spectra again (for warping)
peaks_list <- detectPeaks(all_spectra, method = "MAD", halfWindowSize = 20, SNR = 3)

# Determine warping functions (returns a list of warping functions)
warping_functions <- determineWarpingFunctions(peaks_list, 
                                               reference = reference_peaks,
                                               tolerance = tolerance, 
                                               method = "linear")

# Warp all spectra at once
aligned_spectra <- warpMassSpectra(all_spectra, warping_functions)
dir1_spectra <- aligned_spectra[1:length(spectra_list$DIR1)]
dir2_spectra <- aligned_spectra[(length(spectra_list$DIR1) + 1):length(aligned_spectra)]


#########################################
# Write Aligned Spectra to Separate Folders
#########################################

print("Writing aligned spectra to files...")

for (name in names(aligned_spectra)) {
  aligned_spec <- aligned_spectra[[name]]
  spectraMatrix <- data.frame(mass = mass(aligned_spec), intensity = intensity(aligned_spec))
  
  # Decide which directory to write to based on the source directory of the spectrum
  if (name %in% names(spectra_list$DIR1)) {
    # ECFMAS spectra
    out_filename <- file.path(ECFMAS_OUT_DIR, paste0(name, "_aligned.txt"))
  } else if (name %in% names(spectra_list$DIR2)) {
    # ECFMAR spectra
    out_filename <- file.path(ECFMAR_OUT_DIR, paste0(name, "_aligned.txt"))
  } else {
    # If somehow not found (should not happen) handle error
    warning("Spectrum name not found in either directory lists. Defaulting to ECFMAS_OUT_DIR.")
  }
  
  # Write the file
  file_con <- file(out_filename, open = "wt")
  writeLines(paste("# ", out_filename), file_con)
  writeLines(paste("# ", name), file_con)
  write.table(spectraMatrix, file_con, sep = " ", row.names = FALSE)
  flush(file_con)
  close(file_con)
}

print("All aligned spectra written.")

print("Processing complete!")
sink()
print("Done!")
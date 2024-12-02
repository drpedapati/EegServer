# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pathlib",
#     "scipy",
#     "reportlab",
#     "rich",
#     "seaborn"
# ]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
from pathlib import Path
import traceback
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from io import BytesIO
import logging
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import textwrap
from scipy.signal import savgol_filter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Key constants for frequency range and alpha range
DEFAULT_FREQ_RANGE = (6, 12)
DEFAULT_ALPHA_RANGE = (6, 12)

# Channel mapping and regions
CHANNEL_MAPPING = {
    'F3': 'E24',
    'F4': 'E124',
    'C3': 'E36',
    'C4': 'E104',
    'O1': 'E70',
    'O2': 'E83'
}

REGIONS = {
    'Frontal': ['F3', 'F4'],
    'Central': ['C3', 'C4'],
    'Occipital': ['O1', 'O2']
}

# Constants for PDF generation
PDF_TITLE = "EEG Peak Frequency Analysis Report"
PDF_SUBTITLE = "peakFinderFunction.py at github.com/cincibrainlab"
PDF_CONTACT = "Contact: ernest.pedapati@cchmc.org"
PDF_REFERENCE = (
    "Reference: Dickinson, A., DiStefano, C., Senturk, D., & Jeste, S. S. (2018). "
    "Peak alpha frequency is a neural marker of cognitive function across the autism spectrum. "
    "European Journal of Neuroscience, 47(6), 643-651. "
    "https://doi.org/10.1111/ejn.13645"
)

# Constants for plot titles and labels
PLOT_TITLE_LOG_POWER = "(a) Log Power Spectra with 1/f Trend - {}"
PLOT_TITLE_DETRENDED = "(b) Detrended Spectrum with Peak Fit - {}\nStatus: {}"
PLOT_XLABEL = "Frequency (Hz)"
PLOT_YLABEL_LOG_POWER = "Log Relative Power"
PLOT_YLABEL_DETRENDED = "Detrended Log Relative Power"

# Constants for heatmap
HEATMAP_TITLE = "Peak Frequency Heatmap"
HEATMAP_XLABEL = "Side"
HEATMAP_YLABEL = "Region"
HEATMAP_CBAR_LABEL = "Peak Frequency (Hz)"

# Constants for EGI system information
EGI_INFO_TITLE = "EGI 129 Channel System Information"
EGI_INFO_TEXT = """
Rationale:

The EGI 129 channel system is a high-density EEG system
that provides comprehensive coverage of the scalp.
We focus on key channels that correspond to standard
10-20 system locations:

• Frontal (F3, F4): Involved in executive functions
  and emotional processing
• Central (C3, C4): Associated with sensorimotor
  functions
• Occipital (O1, O2): Primary visual processing areas

This selection allows for comparison between hemispheres
and provides insights into regional brain activity patterns.
"""
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def model_1f_trend(frequencies, powers):
    log_freq = np.log10(frequencies)
    log_power = np.log10(powers)
    slope, intercept, _, _, _ = stats.linregress(log_freq, log_power)
    return slope * log_freq + intercept


def enhanced_smooth_spectrum(spectrum, method='moving_average', **kwargs):
    if method == 'moving_average':
        window_size = kwargs.get('window_size', 3)
        return np.convolve(spectrum, np.ones(window_size)/window_size, mode='same')
    
    elif method == 'gaussian':
        window_size = kwargs.get('window_size', 3)
        sigma = kwargs.get('sigma', 1.0)
        gaussian_window = np.exp(-(np.arange(window_size) - window_size//2)**2 / (2*sigma**2))
        gaussian_window /= np.sum(gaussian_window)
        return np.convolve(spectrum, gaussian_window, mode='same')
    
    elif method == 'savitzky_golay':
        window_length = kwargs.get('window_length', 5)
        poly_order = kwargs.get('poly_order', 2)
        return savgol_filter(spectrum, window_length, poly_order)
    
    elif method == 'median':
        window_size = kwargs.get('window_size', 3)
        return np.array([np.median(spectrum[max(0, i-window_size//2):min(len(spectrum), i+window_size//2+1)]) 
                         for i in range(len(spectrum))])
    
    else:
        raise ValueError("Unsupported smoothing method")

def dickinson_method(frequencies, powers, freq_range, alpha_range, egi_channel=None, standard_channel=None):
    log_powers = np.log10(powers)
    log_trend = model_1f_trend(frequencies, powers)
    detrended_log_powers_raw = log_powers - log_trend
    detrended_log_powers = enhanced_smooth_spectrum(log_powers - log_trend, method='savitzky_golay', window_length=5, poly_order=2)
    #detrended_log_powers = smooth_spectrum(detrended_log_powers)

    alpha_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    alpha_freqs = frequencies[alpha_mask]
    alpha_powers = detrended_log_powers[alpha_mask]

    from rich import print as rprint
    if len(alpha_freqs) == 0:
        return np.nan, None, alpha_freqs, alpha_powers, log_powers, log_trend, detrended_log_powers, "NO_DATA_IN_RANGE"
    
    # Append alpha_powers to a file as a new row
    if egi_channel is not None and standard_channel is not None:
        with open('alpha_powers.txt', 'a') as f:
            row_data = np.concatenate(([standard_channel], detrended_log_powers_raw[alpha_mask]))
            np.savetxt(f, row_data.reshape(1, -1), delimiter=',', fmt='%s', newline='\n')
    
    peaks, _ = find_peaks(alpha_powers, width=1)
    if standard_channel == 'F3':
        rprint(f"peaks for F3: {peaks}")

    if len(peaks) == 0:
        return np.nan, None, alpha_freqs, alpha_powers, log_powers, log_trend, detrended_log_powers, "NO_PEAKS_FOUND"
    peak_prominences = alpha_powers[peaks] - np.min(alpha_powers)
    sorted_peaks = [p for _, p in sorted(zip(peak_prominences, peaks), reverse=True)]

    if standard_channel == 'F3':
        rprint(f"Sorted peaks for F3: {sorted_peaks}")
        rprint(f"Peak prominences for F3: {peak_prominences}")
    # Additional checks for peak validity
    if standard_channel == 'F3':
        rprint(f"Checking peak validity for F3")
    
    for peak_idx in sorted_peaks:
        peak_freq = alpha_freqs[peak_idx]
        if alpha_range[0] <= peak_freq <= alpha_range[1]:
            if standard_channel == 'F3':
                rprint(f"Processing peak at frequency {peak_freq} Hz for F3")
            try:
                p0 = [alpha_powers[peak_idx], peak_freq, .2]
                popt, _ = curve_fit(gaussian, alpha_freqs, alpha_powers, p0=p0, maxfev=1000)
                fitted_curve = gaussian(alpha_freqs, *popt)
                
                if alpha_range[0] <= popt[1] <= alpha_range[1]:
                    if standard_channel == 'F3':
                        rprint(f"Successful fit for F3: peak at {popt[1]} Hz")
                    return popt[1], fitted_curve, alpha_freqs, alpha_powers, log_powers, log_trend, detrended_log_powers, "SUCCESS"
            except RuntimeError:
                if standard_channel == 'F3':
                    rprint(f"Curve fitting failed for F3 at frequency {peak_freq} Hz")
                continue

    alpha_range_mask = (alpha_freqs >= alpha_range[0]) & (alpha_freqs <= alpha_range[1])
    if np.any(alpha_range_mask):
        max_peak_freq = alpha_freqs[alpha_range_mask][np.argmax(alpha_powers[alpha_range_mask])]
        return max_peak_freq, None, alpha_freqs, alpha_powers, log_powers, log_trend, detrended_log_powers, "MAX_PEAK_USED"
    else:
        return np.nan, None, alpha_freqs, alpha_powers, log_powers, log_trend, detrended_log_powers, "NO_VALID_PEAK"

def process_spectrogram(df, freq_range, alpha_range):
    results = []
    if 'freq' not in df.columns:
        raise KeyError("Column 'freq' not found in the DataFrame")
    frequencies = df['freq'].values
    
    for standard_channel, egi_channel in CHANNEL_MAPPING.items():
        column = f'relpow_{egi_channel}'
        if column in df.columns:
            peak_freq, _, _, _, _, _, _, status = dickinson_method(frequencies, df[column].values, freq_range, alpha_range, egi_channel, standard_channel)
            results.append({
                'channel': standard_channel,
                'egi_channel': egi_channel,
                'peak_freq': peak_freq,
                'status': status
            })
    
    return pd.DataFrame(results)

def average_regional_data(df):
    regional_data = {}
    for region, channels in REGIONS.items():
        regional_data[region] = df[[f'relpow_{CHANNEL_MAPPING[ch]}' for ch in channels]].mean(axis=1)
    regional_data['freq'] = df['freq']
    return pd.DataFrame(regional_data)

def plot_spectrogram_analysis(df, channel_or_region, freq_range, alpha_range, is_region=False):
    try:
        frequencies = df['freq'].values
        powers = df[channel_or_region].values if is_region else df[f'relpow_{CHANNEL_MAPPING[channel_or_region]}'].values

        peak_freq, fitted_curve, alpha_freqs, alpha_powers, log_powers, log_trend, detrended_log_powers, status = dickinson_method(frequencies, powers, freq_range, alpha_range)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(frequencies, log_powers, label='Original Spectrum')
        ax1.plot(frequencies, log_trend, '--', label='1/f Trend')
        ax1.set_xlabel(PLOT_XLABEL)
        ax1.set_ylabel(PLOT_YLABEL_LOG_POWER)
        ax1.set_title(PLOT_TITLE_LOG_POWER.format(channel_or_region))
        ax1.legend()
        ax1.set_xlim(0, 30)
        
        ax2.plot(alpha_freqs, alpha_powers, label='Detrended Spectrum')
        if fitted_curve is not None:
            ax2.plot(alpha_freqs, fitted_curve, '--', label='Gaussian Fit')
        if not np.isnan(peak_freq):
            ax2.axvline(x=peak_freq, color='r', linestyle=':', label=f'Peak: {peak_freq:.2f} Hz')
        ax2.set_xlabel(PLOT_XLABEL)
        ax2.set_ylabel(PLOT_YLABEL_DETRENDED)
        ax2.set_title(PLOT_TITLE_DETRENDED.format(channel_or_region, status))
        ax2.legend()
        ax2.set_xlim(freq_range)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_spectrogram_analysis for {channel_or_region}: {str(e)}")
        # Create a figure with error message
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.text(0.5, 0.5, f"Error: {str(e)}\nStatus: {status}", ha='center', va='center')
        ax.axis('off')
        return fig

def load_data(file_path):
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .parquet or .csv files.")
        
def generate_pdf_report(output_path, df, regional_df, freq_range, alpha_range, results):
    logging.info(f"Generating PDF report: {output_path}")
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    def add_page_number(canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawRightString(width - 0.5*inch, 0.5*inch, text)

    # Title Page
    c.setFont("Times-Bold", 18)
    c.drawCentredString(width/2, height - 1.5*inch, PDF_TITLE)

    c.setFont("Times-Roman", 12)
    c.drawCentredString(width/2, height - 2.5*inch, PDF_SUBTITLE)
    c.drawCentredString(width/2, height - 3*inch, PDF_CONTACT)

    intro_text = (
        f"This report presents the results of an EEG peak frequency analysis "
        f"for {os.path.basename(output_path)} using the Dickinson (2018) method. "
        f"It includes individual channel analyses, regional averages, and a "
        f"summary of peak frequencies and their status."
    )
    text_object = c.beginText(1*inch, height - 4*inch)
    text_object.setFont("Times-Roman", 12)
    for line in textwrap.wrap(intro_text, width=70):
        text_object.textLine(line)

    c.drawText(text_object)

    c.setFillColor(HexColor("#4A4A4A"))
    c.setFont("Times-Italic", 9)
    ref_text_object = c.beginText(1*inch, 1*inch)
    for line in textwrap.wrap(PDF_REFERENCE, width=100):
        ref_text_object.textLine(line)
    c.drawText(ref_text_object)

    c.setFillColor(HexColor("#000000"))

    add_page_number(c, None)
    c.showPage()

    # Analysis Parameters Page
    c.setFont("Times-Bold", 14)
    c.drawCentredString(width/2, height - 1*inch, "Analysis Parameters")
    c.setFont("Times-Roman", 12)
    freq_range_str = results['freq_range'].iloc[0]
    alpha_range_str = results['alpha_range'].iloc[0]
    analysis_date = results['analysis_date'].iloc[0]
    c.drawCentredString(width/2, height - 1.5*inch, f"Frequency Range: {freq_range_str}")
    c.drawCentredString(width/2, height - 1.8*inch, f"Alpha Range: {alpha_range_str}")
    c.drawCentredString(width/2, height - 2.1*inch, f"Analysis Date: {analysis_date}")

    # Results Summary Table
    c.setFont("Times-Bold", 11)
    c.drawCentredString(width/2, height - 2.6*inch, "Table 1")
    c.setFont("Times-Roman", 11)
    c.drawCentredString(width/2, height - 2.8*inch, "Results Summary")

    table_data = [
        ['Channel', 'Peak Frequency (Hz)', 'Status']
    ]
    for _, row in results.iterrows():
        table_data.append([
            row['channel'],
            f"{row['peak_freq']:.2f}",
            row['status']
        ])

    table = Table(table_data)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    table.setStyle(table_style)
    table.wrapOn(c, width - inch, height)
    table.drawOn(c, (width - table._width) / 2, height - 4.5*inch - table._height)

    add_page_number(c, None)
    c.showPage()

    # Heatmap Page
    import seaborn as sns
    import matplotlib.pyplot as plt

    channel_order = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    heatmap_data = results.set_index('channel').loc[channel_order, 'peak_freq'].to_frame()

    heatmap_data = heatmap_data.reset_index()
    heatmap_data['Region'] = ['Frontal', 'Frontal', 'Central', 'Central', 'Occipital', 'Occipital']
    heatmap_data['Side'] = ['Left', 'Right'] * 3
    heatmap_data_pivot = heatmap_data.pivot(index='Region', columns='Side', values='peak_freq')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 2]})

    sns.heatmap(heatmap_data_pivot, 
                cmap='YlOrRd', 
                annot=True, 
                fmt='.2f', 
                cbar_kws={'label': 'Peak Frequency (Hz)'},
                vmin=DEFAULT_ALPHA_RANGE[0], vmax=DEFAULT_ALPHA_RANGE[1],
                square=True,
                ax=ax1)

    ax1.set_title('Peak Frequency Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Side', fontsize=12)
    ax1.set_ylabel('Region', fontsize=12)

    ax2.axis('off')
    ax2.set_title('EGI 129 Channel System Information', fontsize=14, fontweight='bold')

    mapping_text = "Electrode Mapping:\n\n"
    mapping_text += "\n".join([f"{k:<3}: {v}" for k, v in CHANNEL_MAPPING.items()])

    rationale_text = "\n\nRationale:\n\n"
    rationale_text += "The EGI 129 channel system is a high-density EEG system\n"
    rationale_text += "that provides comprehensive coverage of the scalp.\n"
    rationale_text += "We focus on key channels that correspond to standard\n"
    rationale_text += "10-20 system locations:\n\n"
    rationale_text += "• Frontal (F3, F4): Involved in executive functions\n"
    rationale_text += "  and emotional processing\n"
    rationale_text += "• Central (C3, C4): Associated with sensorimotor\n"
    rationale_text += "  functions\n"
    rationale_text += "• Occipital (O1, O2): Primary visual processing areas\n\n"
    rationale_text += "This selection allows for comparison between hemispheres\n"
    rationale_text += "and provides insights into regional brain activity patterns."

    ax2.text(0, 1, mapping_text + rationale_text, va='top', ha='left', fontsize=10, linespacing=1.5)

    plt.tight_layout()

    heatmap_buffer = BytesIO()
    plt.savefig(heatmap_buffer, format='png', bbox_inches='tight', dpi=300)
    heatmap_buffer.seek(0)

    plt.close()
    
    fig_width, fig_height = fig.get_size_inches()
    aspect_ratio = fig_width / fig_height

    pdf_image_width = 7 * inch
    pdf_image_height = pdf_image_width / aspect_ratio

    c.drawImage(ImageReader(heatmap_buffer), (width - pdf_image_width) / 2, height - pdf_image_height - 0.5*inch, width=pdf_image_width, height=pdf_image_height)
    # Figure Description
    c.setFont("Times-Bold", 11)
    c.drawString(1*inch, 1.5*inch, "Figure 1")
    c.setFont("Times-Roman", 11)
    c.drawString(1*inch, 1.3*inch, "Peak Frequency Heatmap and EGI 129 Channel System Information")
    
    c.setFont("Times-Roman", 10)
    plot_description = (
        "For each channel and region, two plots are provided:\n"
        "(a) Log Power Spectra with 1/f Trend: Shows the original spectrum and the fitted 1/f trend.\n"
        "(b) Detrended Spectrum with Peak Fit: Displays the detrended spectrum and the Gaussian fit used to identify the peak frequency."
    )
    text_object = c.beginText(1*inch, 1.0*inch)
    text_object.setFont("Times-Roman", 10)
    for line in plot_description.split('\n'):
        text_object.textLine(line)
    c.drawText(text_object)

    add_page_number(c, None)
    c.showPage()

    # Channel and Regional Plots
    y_position = height - 1*inch

    for channel in CHANNEL_MAPPING.keys():
        logging.info(f"Processing channel: {channel}")
        try:
            fig = plot_spectrogram_analysis(df, channel, DEFAULT_FREQ_RANGE, DEFAULT_ALPHA_RANGE)
            img_data = BytesIO()
            fig.savefig(img_data, format='png')
            img_data.seek(0)
            c.drawImage(ImageReader(img_data), 1*inch, y_position - 4*inch, width=6*inch, height=4*inch)
            plt.close(fig)

            c.setFont("Helvetica-Bold", 12)
            c.drawString(1*inch, y_position, f"Figure: Channel {channel}")
            y_position -= 4.5*inch

            if y_position < 2*inch:
                add_page_number(c, None)
                c.showPage()
                y_position = height - 1*inch
        except Exception as e:
            logging.error(f"Error processing channel {channel}: {str(e)}")
            c.setFont("Helvetica", 10)
            c.drawString(1*inch, y_position, f"Error processing channel {channel}: {str(e)}")
            y_position -= 0.5*inch

    add_page_number(c, None)
    c.showPage()
    y_position = height - 1*inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, y_position, "Regional Averages")
    y_position -= 0.5*inch

    for region in REGIONS.keys():
        logging.info(f"Processing region: {region}")
        try:
            fig = plot_spectrogram_analysis(regional_df, region, DEFAULT_FREQ_RANGE, DEFAULT_ALPHA_RANGE, is_region=True)
            img_data = BytesIO()
            fig.savefig(img_data, format='png')
            img_data.seek(0)
            c.drawImage(ImageReader(img_data), 1*inch, y_position - 4*inch, width=6*inch, height=4*inch)
            plt.close(fig)

            c.setFont("Helvetica-Bold", 12)
            c.drawString(1*inch, y_position, f"Figure: {region} Region")
            y_position -= 4.5*inch

            if y_position < 2*inch:
                add_page_number(c, None)
                c.showPage()
                y_position = height - 1*inch
        except Exception as e:
            logging.error(f"Error processing region {region}: {str(e)}")
            c.setFont("Helvetica", 10)
            c.drawString(1*inch, y_position, f"Error processing region {region}: {str(e)}")
            y_position -= 0.5*inch

    add_page_number(c, None)
    c.save()
    buffer.seek(0)
    with open(output_path, 'wb') as f:
        f.write(buffer.getvalue())
    logging.info(f"PDF report generated successfully: {output_path}")

def process_eeg_file(input_file, output_dir, freq_range=DEFAULT_FREQ_RANGE, alpha_range=DEFAULT_ALPHA_RANGE):
    try:
        df = load_data(input_file)
        results = process_spectrogram(df, freq_range=freq_range, alpha_range=alpha_range)
        
        # Add metadata to results
        results['source_file'] = os.path.basename(input_file)
        results['analysis_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        results['freq_range'] = f"{freq_range[0]}-{freq_range[1]}"
        results['alpha_range'] = f"{alpha_range[0]}-{alpha_range[1]}"
        
        # Save results
        output_csv = os.path.join(output_dir, f"{Path(input_file).stem}_peak_freq_results.csv")
        results.to_csv(output_csv, index=False)
        
        # Generate PDF report
        regional_df = average_regional_data(df)
        pdf_output = os.path.join(output_dir, f"{Path(input_file).stem}_peak_freq_report.pdf")
        generate_pdf_report(pdf_output, df, regional_df, freq_range, alpha_range, results)
        
        return True, f"Processed {input_file} successfully. Results saved to {output_csv} and {pdf_output}"
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}\n{traceback.format_exc()}")
        return False, f"Error processing {input_file}: {str(e)}\n{traceback.format_exc()}"

def batch_process_eeg_files(input_dir, output_dir, freq_range=DEFAULT_FREQ_RANGE, alpha_range=DEFAULT_ALPHA_RANGE, file_filter='_spectro.csv'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    from rich import print as rprint
    from rich.console import Console
    console = Console()

    rprint(f"[bold green]Input directory:[/bold green] {input_dir}")
    rprint(f"[bold green]Output directory:[/bold green] {output_dir}")
    
    results = []
    for file in os.listdir(input_dir):
        if file.endswith(file_filter):
            input_file = os.path.join(input_dir, file)
            success, message = process_eeg_file(input_file, output_dir, freq_range, alpha_range)
            results.append({'file': file, 'success': success, 'message': message})
            if success:
                console.print(f"✅ [bold cyan]{file}[/bold cyan] processed successfully! 🎉")
            else:
                console.print(f"❌ [bold red]{file}[/bold red] processing failed. 😞")
    
    return pd.DataFrame(results)

# Example usage:
# batch_results = batch_process_eeg_files('/path/to/input/dir', '/path/to/output/dir')
# print(batch_results)
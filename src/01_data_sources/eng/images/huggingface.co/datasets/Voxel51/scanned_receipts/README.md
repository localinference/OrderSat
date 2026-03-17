---
annotations_creators: []
language: en
size_categories:
  - n<1K
task_categories:
  - object-detection
  - visual-question-answering
  - visual-document-retrieval
task_ids: []
pretty_name: icdar-sroie-train
tags:
  - fiftyone
  - image
  - object-detection
dataset_summary: >




  This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 712
  samples.


  ## Installation


  If you haven't already, install FiftyOne:


  ```bash

  pip install -U fiftyone

  ```


  ## Usage


  ```python

  import fiftyone as fo

  from fiftyone.utils.huggingface import load_from_hub


  # Load the dataset

  # Note: other available arguments include 'max_samples', etc

  dataset = load_from_hub("Voxel51/scanned_receipts")


  # Launch the App

  session = fo.launch_app(dataset)

  ```
license: cc-by-4.0
---

# Dataset Card for Scanned Receipts OCR and Information Extraction

![image/png](icdar_sroie.gif)

This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 712 samples.

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/scanned_receipts")

# Launch the App
session = fo.launch_app(dataset)
```

### Dataset Description

The **ICDAR-SROIE (Scanned Receipts OCR and Information Extraction)** dataset comprises **1,000 whole scanned receipt images** collected from real-world scenarios. This dataset was introduced as part of the **ICDAR 2019 Competition** to advance research in document analysis, optical character recognition (OCR), and information extraction from structured documents.

The dataset supports three interconnected tasks:

1. **Scanned Receipt Text Localization**: Accurately localizing text regions in receipt images
2. **Scanned Receipt OCR**: Recognizing and transcribing text content from detected regions
3. **Key Information Extraction**: Extracting structured information (company, address, date, total) from receipts

The receipts originate primarily from shops, stores, and restaurants, representing diverse real-world formats, layouts, and printing qualities. This diversity makes it an excellent benchmark for evaluating robust document understanding systems.

**Key Characteristics:**

- **Total Images**: 1,000 scanned receipt images
- **Training Set**: 712 images with annotations
- **Test Set**: 347 images (with 361 in some versions)
- **Image Format**: JPEG
- **Languages**: Primarily English, with some multilingual content
- **Real-world Data**: Authentic receipts with natural variations in quality, layout, and format

**Curated by:** Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, C.V. Jawahar  
**Funded by:** ICDAR 2019 Competition organizers  
**Language(s)**: Primarily English (en) with multilingual elements

### Dataset Sources

- **Repository**: [ICDAR-SROIE Competition Page](https://rrc.cvc.uab.es/?ch=13)
- **Paper**: "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction" (ICDAR 2019)
- **Alternative Access**: [Papers with Code - SROIE Dataset](https://paperswithcode.com/dataset/sroie)

### Supported Tasks and Leaderboards

#### Task 1: Scanned Receipt Text Localization

Detect and localize all text regions in receipt images using quadrilateral bounding boxes.

**Evaluation Metrics**: Precision, Recall, F1-Score (IoU threshold-based)

#### Task 2: Scanned Receipt OCR

Transcribe text from detected regions into machine-readable format.

**Evaluation Metrics**: Word-level and character-level accuracy

#### Task 3: Key Information Extraction

Extract four key fields from receipts:

- **Company**: Business/merchant name
- **Address**: Business address
- **Date**: Transaction date
- **Total**: Total transaction amount

**Evaluation Metrics**: Exact match accuracy for each field

### Dataset Information Summary

The ICDAR-SROIE dataset contains scanned receipt images with:

- **Text bounding boxes**: Quadrilateral bounding boxes (4 vertices) with coordinates in clockwise order starting from the top
- **Text transcripts**: The OCR text for each bounding box
- **Metadata**: Key information extracted from receipts including:
  - Company name
  - Address
  - Date
  - Total amount

## Dataset Structure

Each image in the dataset has associated annotation files:

```
X00016469612.jpg              # Receipt image
X00016469612_bbox.txt         # Text bounding boxes and transcripts
X00016469612_metadata.txt     # Extracted key information (JSON)
```

### Bounding Box Format

Each line in the `_bbox.txt` file contains:

```
x1,y1,x2,y2,x3,y3,x4,y4,transcript
```

Where `(x1,y1)`, `(x2,y2)`, `(x3,y3)`, `(x4,y4)` are the four vertices of the bounding box in clockwise order.

### Metadata Format

The `_metadata.txt` file contains JSON with extracted information:

```json
{
  "company": "STARBUCKS STORE #10208",
  "address": "11302 EUCLID AVENUE, CLEVELAND, OH (216) 229-0749",
  "date": "14/03/2015",
  "total": "4.95"
}
```

## FiftyOne Dataset Schema

The parsed dataset includes the following fields:

- **filepath**: Path to the image file
- **metadata**: Image metadata (width, height, etc.)
- **company**: Company name extracted from receipt
- **date**: Date on the receipt
- **address**: Address from the receipt
- **total**: Total amount on the receipt
- **text_detections**: Bounding box detections (axis-aligned rectangles)
  - Each detection has the transcript as its label
  - Bounding boxes are in relative coordinates [x, y, width, height]
- **text_polygons**: Original 4-point polygon annotations
  - Preserves the exact quadrilateral shape of text regions
  - Useful for rotated or perspective-distorted text

```
Sample fields:
  - id: <fiftyone.core.fields.ObjectIdField>
  - filepath: <fiftyone.core.fields.StringField>
  - tags: <fiftyone.core.fields.ListField>
  - metadata: <fiftyone.core.fields.EmbeddedDocumentField>
  - company: <fiftyone.core.fields.StringField>
  - date: <fiftyone.core.fields.StringField>
  - address: <fiftyone.core.fields.StringField>
  - total: <fiftyone.core.fields.StringField>
  - text_detections: <fiftyone.core.fields.EmbeddedDocumentField>
  - text_polygons: <fiftyone.core.fields.EmbeddedDocumentField>

```

## Visualization

Once the dataset is loaded in FiftyOne, you can:

1. **Browse images**: Navigate through all receipt images
2. **View text detections**: See bounding boxes overlaid on images
3. **Inspect polygons**: View the original 4-point annotations
4. **Filter by metadata**: Search for specific companies, dates, or amounts
5. **Export selections**: Save filtered subsets for further analysis

### FiftyOne App Features

- Toggle between `text_detections` (rectangles) and `text_polygons` (quadrilaterals)
- Filter samples by company, date range, or total amount
- View text transcripts by hovering over bounding boxes
- Create custom views and tags
- Export annotations in various formats

## Use Cases

### Direct Use

The ICDAR-SROIE dataset is intended for:

1. **OCR Model Development**: Training and evaluating text detection and recognition models on real-world document images
2. **Information Extraction Research**: Developing systems that extract structured information from semi-structured documents
3. **Document Understanding**: Building end-to-end document analysis pipelines that combine localization, recognition, and extraction
4. **Benchmark Evaluation**: Comparing the performance of different OCR and IE approaches on standardized data
5. **Transfer Learning**: Pre-training models on receipt data for adaptation to other document types
6. **Data Augmentation Studies**: Experimenting with augmentation techniques for document images
7. **Multi-task Learning**: Training models that jointly optimize for detection, recognition, and extraction

**Recommended Use Cases:**

- Academic research in computer vision and NLP
- Development of commercial OCR systems (subject to licensing)
- Educational projects for learning document AI
- Prototyping receipt digitization applications

## Dataset Creation

### Curation Rationale

The ICDAR-SROIE dataset was created to address the growing need for robust OCR and information extraction systems capable of handling real-world scanned documents. Receipts were chosen as the target document type because they:

1. **Represent real-world challenges**: Variable layouts, print quality issues, scanning artifacts
2. **Have practical applications**: Expense tracking, accounting automation, retail analytics
3. **Require multi-level understanding**: From pixel-level text detection to semantic field extraction
4. **Are widely available**: Common documents that facilitate data collection
5. **Have structured information**: Enable evaluation of extraction accuracy

The competition format encouraged development of complete end-to-end systems rather than isolated components.

### Source Data

#### Data Collection and Processing

**Collection Process:**

- Receipts were collected from real-world transactions at various retail establishments
- Images were captured using scanners and mobile devices to reflect practical use cases
- Sources included shops, restaurants, and service providers
- Collection focused on ensuring diversity in layouts, formats, and printing styles

**Processing Steps:**

1. **Image Standardization**: Receipts were scanned and converted to JPEG format
2. **Quality Control**: Images were reviewed for readability and completeness
3. **Annotation**: Expert annotators marked text bounding boxes and transcribed content
4. **Validation**: Annotations were validated for accuracy and consistency
5. **Key Information Extraction**: Four key fields (company, address, date, total) were manually extracted
6. **Format Conversion**: Annotations were stored in accessible text formats

**Tools and Methods:**

- Manual annotation of bounding boxes using annotation software
- Multiple annotator validation for quality assurance
- Standardized guidelines for consistent annotation

#### Who are the Source Data Producers?

**Data Sources:**

- Real receipts from actual business transactions
- Establishments primarily located in Asia (based on business names and languages observed)
- Mix of local shops, international chains, and restaurants
- Time period: Receipts from approximately 2015-2018 based on dates in metadata

**Collectors:**

- Dataset curated by research teams from multiple institutions
- Part of ICDAR 2019 competition organization efforts
- No personally identifiable information of customers was intentionally collected

### Annotations

#### Annotation Process

**Text Localization (Task 1):**

- Annotators manually drew quadrilateral bounding boxes around each text instance
- Vertices marked in clockwise order starting from the top-left
- Boxes follow text orientation (can be rotated for angled text)
- Approximately 30-50 text regions per receipt on average

**OCR Transcription (Task 2):**

- Each bounded text region was manually transcribed
- Transcriptions preserve original text including punctuation and special characters
- Multi-line text blocks were typically annotated as separate instances
- Quality control included cross-validation by multiple annotators

**Key Information Extraction (Task 3):**

- Four fields manually extracted: company, address, date, total
- JSON format for structured storage
- Guidelines provided for handling edge cases (multiple totals, missing information)
- Consistency checks performed across the dataset

**Annotation Guidelines:**

- Detailed instructions provided to ensure consistency
- Inter-annotator agreement measured and discrepancies resolved
- Estimated annotation time: 10-15 minutes per receipt

#### Who are the Annotators?

The annotations were created by:

- Trained human annotators with expertise in document analysis
- Researchers and competition organizers
- Quality control performed by domain experts
- Multi-stage validation process to ensure accuracy

**Annotator Demographics:**

- Information about specific annotator demographics is not publicly available
- Likely included researchers familiar with OCR and document understanding
- Native language speakers for multilingual content validation

#### Personal and Sensitive Information

**Privacy Considerations:**

The dataset contains real-world receipts which may include:

- ✅ **Business Names**: Company names are public information
- ✅ **Business Addresses**: Public business addresses
- ✅ **Transaction Dates**: Non-sensitive temporal information
- ✅ **Transaction Amounts**: Individual transaction totals

The dataset does NOT intentionally contain:

- ❌ Customer names
- ❌ Credit card numbers
- ❌ Personal phone numbers or email addresses
- ❌ Customer addresses

**Anonymization:**

- Receipts were selected/processed to exclude personal customer information
- Any customer-identifying information visible should be considered incidental
- Business information (company names, addresses) is inherently public

**Usage Recommendations:**

- Researchers should not attempt to identify individuals from this data
- Any incidentally visible personal information should not be extracted or shared
- Follow ethical guidelines for research with real-world data

## Citation

If you use the ICDAR-SROIE dataset in your research, please cite the original competition paper:

### BibTeX

```bibtex
@article{huang2021icdar2019,
  title     = {ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction},
  author    = {Zheng Huang and Kai Chen and Jianhua He and Xiang Bai and Dimosthenis Karatzas and Shijian Lu and C. V. Jawahar},
  journal   = {arXiv preprint arXiv:2103.10213},
  year      = {2021},
  doi       = {10.48550/arXiv.2103.10213},
  url       = {https://arxiv.org/abs/2103.10213}
}

```

### APA

Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., & Jawahar, C. V. (2019). ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_ (pp. 1516-1520). IEEE.

## References

### Dataset Resources

- **Competition Page**: [ICDAR-SROIE Challenge](https://rrc.cvc.uab.es/?ch=13)
- **Papers with Code**: [SROIE Dataset](https://paperswithcode.com/dataset/sroie)
- **Leaderboard**: [ICDAR 2019 SROIE Results](https://rrc.cvc.uab.es/?ch=13&com=evaluation)

### Related Tools

- **FiftyOne**: [Documentation](https://docs.voxel51.com/)
- **FiftyOne Polylines**: [Guide](https://docs.voxel51.com/user_guide/using_datasets.html#polylines)
- **FiftyOne GitHub**: [Repository](https://github.com/voxel51/fiftyone)

### Related Papers

- [Huang et al. (2019) - ICDAR2019 Competition paper](https://arxiv.org/abs/2103.10213)
- Various competition participant papers describing different approaches

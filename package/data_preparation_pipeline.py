"""
class for data preparation pipeline for spam-vs-ham classification task
"""
import os
import sys
import shutil
import re
import csv
import base64
import numpy as np
import quopri
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

'''
add home directory (Pydata-Book) to env variables 
'''
home_directory = os.path.abspath('../../')
sys.path.append(home_directory)

# utility functions
from Utils.data_utils import download_multiple_files_from_url


class SpamAssassinDataPreparation:
    def __init__(self, urls=None, override_prev_data=False):
        if urls is None:
            urls = []

        self.urls = urls
        self.override_prev_data = override_prev_data

        self.downloaded_data_dir = os.path.abspath('./dataset/downloaded-data')
        self.data_csv_file = os.path.abspath('./dataset/data.csv')
        self.keyword_txt_file = os.path.abspath('./dataset/keywords.txt')
        self.combined_data_dir = os.path.abspath('./dataset/combined-data')
        self.spam_samples_dir = os.path.join(self.downloaded_data_dir, 'spam')

        self.spam_email_keywords = None

        self.manual_stop_words = [
            '20', 'nbsp', 'com', 'text', 'encoding', 'content', 
            'type', 'transfer', 'encoding', 'base64', 
            'charset', '8859', 'iso', 'linux', 'don', 'plain', 'subject',
            'html', 'does', 'know', 've', 'em', 'let', 'jm', 'amp', 
            'pgh0bww', 'pgi', 'quot', 'cd', 'got', 'ilug', 'hi', 'set',
            'pha', 'cn', '_______________________________________________',
            'lt', 'did', 'da', 'aa', 'o1', 'oe', 'pgzvbnqg', 'ce', 'ne', 'a1',
            'non', 'dr', 'ca', 'ffffff99', 'al', 'sf', 'pa', '2e', 'wa', 'rom',
            'href', 'fed', 'se', 'en', 'ia', 'ao', 'fl', 'iii', 'wi', 'base'
        ]

        self.is_new_data_available_for_cleaning = False
        self.is_new_data_available_for_keyword_preparation = False

    @staticmethod
    def create_dir_if_not_exists(dir_path):
        if not os.path.exists(dir_path):
            print(f'LOG: {dir_path} directory does not exists, creating...')
            os.makedirs(dir_path)

    @staticmethod
    def remove_numbers(text, replacement_text=""):
        pattern = re.compile(r'\d+')
        
        text_without_numbers = pattern.sub(replacement_text, text)
        return text_without_numbers

    @staticmethod
    def remove_urls(text, replacement_text=""):
        # Define a regex pattern to match URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')

        # Use the sub() method to replace URLs with the specified replacement text
        text_without_urls = url_pattern.sub(replacement_text, text)

        return text_without_urls

    @staticmethod
    def decode_quoted_printable_text(text):
        try:
            # Attempt to decode quoted-printable encoding
            decoded_text = quopri.decodestring(text.encode()).decode()
        except Exception as e:
            # If decoding fails, assume the text is not encoded and use it directly
            decoded_text = text

        return decoded_text

    @staticmethod
    def remove_html_tags(text):
        html_pattern = re.compile(r'<[^>]+>')

        return html_pattern.sub('', text)

    @staticmethod
    def decode_base64(match):
        decoded_bytes = base64.b64decode(match.group(0))

        try:
            decoded_string = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            decoded_string = decoded_bytes.decode('latin-1')

        return decoded_bytes

    # -----------------------------------------------------

    def perform_func_on_each_file(self, func, read_in_byte_mode=False):
        for root, dirs, files in os.walk(self.downloaded_data_dir):
            base_dir_name = os.path.basename(root)

            print(f'LOG: processing files of {base_dir_name} dir')

            for file_item in files:
                file_path = os.path.join(root, file_item)

                try:
                    with open(file_path, 'rb' if read_in_byte_mode else 'r') as file:
                        # pass file to function
                        func(file, file_path)

                except UnicodeDecodeError:
                    print(f'LOG: some error during reading the contents of file {file_path}')
                    continue

    def replace_base64_encoding(self, file, file_path):
        content = file.read()
        # regex pattern to find Base64 encoded strings
        pattern = re.compile(rb'(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)')

        matches = pattern.findall(content.encode())

        if not matches:
            print(f'LOG: no base_64 encoded string found, skipping the file: {file_path}')
            return
        else:
            print(f'found base_64: {len(matches)}')

        try:
            # decoded_content = pattern.sub(self.decode_base64, content.encode())
            decoded_content = re.sub(pattern, self.decode_base64, content.encode())
        except TypeError:
            # in case of non-binary files return without any changes
            print(f'LOG: {file_path} is non-binary')
            return

        # Write the decoded content back to the file
        with open(file_path, 'wb') as file:
            try:
                file.write(decoded_content)
            except Exception as e:
                print(f'ERROR: some error occurred in writing file: {file_path}: {str(e)}')

    def clean_text_data(self, text):
        content_split = text.split('\n\n', 1)

        if len(content_split) > 1:
            # If there are two parts, save the content
            clean_text = content_split[1]
        else:
            # In case where there is no header
            clean_text = text

        clean_text = self.remove_numbers(clean_text)
        clean_text = self.remove_urls(clean_text)
        clean_text = self.decode_quoted_printable_text(clean_text)
        clean_text = self.remove_html_tags(clean_text)

        return clean_text

    def preprocess_file_content(self, file, file_path):
        file_content = file.read()

        # clean file content
        email_body = self.clean_text_data(file_content)

        # Write the cleaned text back to the file
        with open(file_path, 'w') as file:
            try:
                file.write(email_body)
            except Exception:
                print(f'ERROR: some error occurred in writing file: {file_path}')

    def create_keyword_frequency_dict(self, content):
        row = {}

        # Try to get keywords first from instance var, if not present try looking into saved keyword file
        keywords = self.spam_email_keywords if self.spam_email_keywords else self.read_keyword_from_txt_file()

        # If both of the above methods fail raise exception
        if not keywords:
            raise RuntimeError('ERROR: Spam keywords are not available')

        for keyword in keywords:
            matches = re.findall(keyword.lower(), content.lower())

            row[keyword] = len(matches)

        return row

    def save_keywords_to_txt_file(self, keywords):
        with open(self.keyword_txt_file, 'w') as file:
            file.write(','.join(keywords))

    def read_keyword_from_txt_file(self):
        try:
            with open(self.keyword_txt_file, 'r') as file:
                words = file.read().split(',')

                return words
        except FileNotFoundError:
            return None

    """
    --------------------------------------------------------
    
    Data preparation steps
    """

    def download_files(self):
        if os.path.isdir(self.downloaded_data_dir):  # if exist
            if self.override_prev_data or (not os.listdir(self.downloaded_data_dir)):  # If empty
                # Then only download data again
                download_multiple_files_from_url(self.urls, self.downloaded_data_dir, use_tarfile=True)

                # If new data is downloaded set flag such that this data can be cleaned
                self.is_new_data_available_for_cleaning = True
                self.is_new_data_available_for_keyword_preparation = True
            else:
                print("LOG: Data is already present")

    def clean_email_file_content(self):
        if self.is_new_data_available_for_cleaning:
            # Clean content of each file
            self.perform_func_on_each_file(self.preprocess_file_content)
            self.is_new_data_available_for_cleaning = False
        else:
            print('LOG: Email data is already clean.')

    def select_spam_keywords(self, verbose=False):
        if not self.is_new_data_available_for_keyword_preparation:
            print('LOG: keywords already prepared')
            return

        corpus = []

        for filename in os.listdir(self.spam_samples_dir):
            file_path = os.path.join(self.spam_samples_dir, filename)
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
                    corpus.append(text)
            except UnicodeDecodeError:
                print(f'LOG: some error during reading the contents of file {file_path}')
                continue

        stop_words = ENGLISH_STOP_WORDS.union(self.manual_stop_words)

        # Create a TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            sublinear_tf=True,
            max_df=0.5,
            min_df=5,
            stop_words=list(stop_words)
        )

        # Fit the vectorizer to the corpus and transform the corpus into a TF-IDF matrix
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

        # Get the feature names (words) from the vectorizer
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # total TF-IDF for all files
        total_tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).squeeze()

        # Get the INDICES of the terms sorted by their aggregated TF-IDF scores
        sorted_indices_total = total_tfidf_scores.argsort()[::-1]  # reversed (sorting in descending order)

        # -- extract keywords according to TF-IDF score -- #
        keywords = []

        for i in range(len(sorted_indices_total)):
            word_idx_total = sorted_indices_total[i]
            word_total = feature_names[word_idx_total]

            keywords.append(word_total)

            if verbose:
                tfidf_score_total = total_tfidf_scores[word_idx_total]
                print(f"{word_total}: {tfidf_score_total}")

        self.spam_email_keywords = keywords

        # save keywords to text file, for use in prediction
        self.save_keywords_to_txt_file(keywords)

        print(f"LOG: {len(keywords)} keywords are extracted from spam examples")

    def walk_and_create_csv_of_spam_keywords(self):
        if not self.override_prev_data and os.path.exists(self.data_csv_file):
            print('LOG: Override not permitted, data csv already exists')
            return self.data_csv_file

        with open(self.data_csv_file, 'w') as csv_file:
            csv_headers = self.spam_email_keywords[:]
            csv_headers.append('Label')

            writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
            writer.writeheader()

            # iterate over each file to find matching keywords
            # Todo: replace following walk with perform_func_on_each_file static function
            for root, dirs, files in os.walk(self.downloaded_data_dir):
                base_dir_name = os.path.basename(root)

                print(f'LOG: working on files of {base_dir_name} dir')

                is_spam = 'spam' in base_dir_name  # isSpam -> label = 'spam'

                for file in files:
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r') as email_file:
                            file_content = email_file.read()

                            row = self.create_keyword_frequency_dict(file_content)
                            
                            # add label to row
                            # row['Label'] = 'spam' if isSpam else 'ham'
                            row['Label'] = is_spam

                            # write new row to csv file
                            writer.writerow(row)
                    except UnicodeDecodeError:
                        print(f'LOG: some error during reading the contents of file {file_path}')
                        continue

        return self.data_csv_file

    def run_data_processing_pipeline(self):
        self.download_files()
        print('All files downloaded'.center(30, '-'), end='\n\n')
        self.clean_email_file_content()
        print('All files are cleaned'.center(30, '-'), end='\n\n')
        self.select_spam_keywords()
        print('Spam keywords prepared'.center(30, '-'), end='\n\n')
        self.walk_and_create_csv_of_spam_keywords()
        print('data CSV prepared'.center(30, '-'), end='\n\n')

        return self.data_csv_file

    def run_prediction_data_processing_pipeline(self, email_content):
        clean_content = self.clean_text_data(email_content)
        keyword_dict = self.create_keyword_frequency_dict(clean_content)

        return keyword_dict

    # !! currently unused
    def decode_base64_encoded_files(self):
        if self.is_new_data_available_for_cleaning:
            # Clean content of each file
            self.perform_func_on_each_file(self.replace_base64_encoding)
        else:
            print('LOG: Email data is already base64 decoded.')

    # !! currently unused
    def move_files_to_combined_folder(self):

        # Create combined-data dir
        self.create_dir_if_not_exists(self.combined_data_dir)

        # If data already present and override not allowed
        if os.listdir(self.combined_data_dir) and not self.override_prev_data:
            print('LOG: Override not permitted, combined data already present')
            return

        for root, dirs, files in os.walk(self.downloaded_data_dir):

            print(f'LOG: Moving {len(files)} files from {root} dir')

            for file in files:
                # Construct the source and destination paths for each file
                source_path = os.path.join(root, file)
                destination_path = os.path.join(self.combined_data_dir, file)

                # Move the file to the destination directory
                shutil.move(source_path, destination_path)

                # print(f"LOG: Moved {source_path} to {destination_path}")

        print("LOG: All download dir files moved successfully.")

"""
Tested with Python 3.8.5

Sample processing tool:

- Takes a directory as input
- and a path to the output file to produce
- then classifies each image
- and produces the correct CSV output

=> Just update the "MyClassifier" methods!
"""

from argparse import ArgumentParser
import os
import os.path
from typing import Dict, List

import numpy as np
import cv2
import joblib

class MyClassifier:
    """Sample classifier class to update.
    """
    def hu_moments(img):
        """
        Computes the 7 hu moments of an image
        """
        def grayscale(img):
            """
            Turn an rgb image into grayscale image
            """
            return np.dot(img[..., :3], [.299, .587, .114])

        def inverted(img):
            """
            Invert all pixel values of an image
            """
            return np.full(img.shape, 255) - img

        def raw_moment(i, j, img):
            """
            Computes raw moment i j of an image
            """
            moment = 0
            for y in range(len(img)):
                for x in range(len(img[0])):
                    moment += x ** i * y ** j * img[y][x]
            return moment

        def central_moment(centroid, i, j, img):
            """
            Computes central moment i j of an image
            Given the centroid for optimization reasons
            """
            moment = 0
            for y in range(len(img)):
                for x in range(len(img[0])):
                    moment += (x - centroid[0]) ** i * (y - centroid[1]) ** j * img[y][x]
            return moment

        def scale_invariant_moment(centroid, c0, i, j, img):
            """
            Computes scale invariant moment i j of an image
            Given the centroid and the central moment 0 0 for optimization reasons
            """
            return central_moment(centroid, i, j, img) / c0 ** (1 + (i + j) / 2)

        img_gray = inverted(grayscale(img))

        avg = raw_moment(0, 0, img_gray)
        centroid = (raw_moment(1, 0, img_gray) / avg, raw_moment(0, 1, img_gray) / avg)

        c0 = central_moment(centroid, 0, 0, img_gray)

        nu20 = scale_invariant_moment(centroid, c0, 2, 0, img_gray)
        nu11 = scale_invariant_moment(centroid, c0, 1, 1, img_gray)
        nu02 = scale_invariant_moment(centroid, c0, 0, 2, img_gray)
        nu30 = scale_invariant_moment(centroid, c0, 3, 0, img_gray)
        nu21 = scale_invariant_moment(centroid, c0, 2, 1, img_gray)
        nu12 = scale_invariant_moment(centroid, c0, 1, 2, img_gray)
        nu03 = scale_invariant_moment(centroid, c0, 0, 3, img_gray)

        hu = np.zeros(7)

        t0 = nu30 + nu12;
        t1 = nu21 + nu03;
        q0 = t0 * t0
        q1 = t1 * t1;
        n4 = 4 * nu11;
        s = nu20 + nu02;
        d = nu20 - nu02;

        hu[0] = s;
        hu[1] = d * d + n4 * nu11;
        hu[3] = q0 + q1;
        hu[5] = d * (q0 - q1) + n4 * t0 * t1;

        t0 *= q0 - 3 * q1;
        t1 *= 3 * q0 - q1;
        q0 = nu30 - 3 * nu12;
        q1 = 3 * nu21 - nu03;

        hu[2] = q0 * q0 + q1 * q1;
        hu[4] = q0 * t0 + q1 * t1;
        hu[6] = q1 * t0 - q0 * t1;

        return hu
    
    def reduce_color(img, kmeans): 
        """
        Reduce the colors in an image

        Parameters
        ----------
        img: image
            Reference image
        kmeans: Kmeans
            kmeans used to determine new colors
        """
        prediction = kmeans.predict(img.reshape(len(img) * len(img[0]), 3))
        histo = np.zeros(len(kmeans.cluster_centers_))
        for i in range(len(img)): 
            for j in range(len(img[0])):
                if img[i, j][0] != 255 or img[i, j][1] != 255 or img[i, j][2] != 255:
                    histo[prediction[i * len(img[0]) + j]] += 1
        return histo

    def normalize_histogram(histo): 
        """
        Normalize histogram

        Parameter
        ----------
        histo: [int]
            image histogram
        """
        nbPixels = np.sum(histo)
        for i in range(len(histo)): 
            if histo[i] != 0:
                histo[i] = histo[i] / nbPixels
        return histo
    
    def __init__(self) -> None:
        self.clf_hu = joblib.load('hu.clf')
        self.clf_col = joblib.load('col.clf')
        self.clf_kmcol = joblib.load('km_col.clf')

    def classify_image(self, image_path: str) -> int:
        """Classify the file for which the path is given,
        returning the correct class as output (1-56) or -1 to reject

        Args:
            image_path (str): Full path to the image to process

        Returns:
            int: Class of the symbol contained (1-56 or -1).

        WARNING: `0` is not a valid class here.
                 You may need to adjust your classifier outputs (typically 0-55).
        """
        img = cv2.imread(image_path);
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        
        hm = [MyClassifier.hu_moments(img)]
        histo = [MyClassifier.normalize_histogram(MyClassifier.reduce_color(img, self.clf_kmcol))]
        
        res_hu = self.clf_hu.predict_proba(hm)
        res_col = self.clf_col.predict_proba(histo)
        res = res_col + res_hu, self.clf_hu.classes_

        return int(res[1][res[0].argmax()])


def save_classifications(image_classes: Dict[str,int], output_path: str):
    """Save classification results to a CSV file

    Args:
        image_classes (Dict[str,int]): dict of base filename -> class id
        output_path (str): will store a CSV in the following format:
    ```csv
    filename,class_id
    filename,class_id
    (no header, accept any leading 0 for class_id)
    ```
    """
    with open(output_path, 'w', encoding="utf8") as out_file:
        for filename, cls_id in image_classes.items():
            out_file.write(f"{filename},{cls_id:02d}\n")


def find_png_files_in_dir(input_dir_path: str) -> List[str]:
    """Returns a list of PNG files contained in a directory, without any leading directory.

    Args:
        input_dir_path (str): Path to directory

    Returns:
        List[str]: List of PNG files (no leading directory).
    """
    with os.scandir(input_dir_path) as entry_iter:
        result = [entry.name for entry in entry_iter if entry.name.endswith('.png') and entry.is_file()]
        return result


def main():
    """Main function."""
    # CLI
    parser = ArgumentParser(description="Sample processing program.")
    parser.add_argument("--test_dir", required=True,
                        help="Path to the directory containing the test files.")
    parser.add_argument("--output", required=True,
                        help="Path to output CSV file.")
    args = parser.parse_args()

    # Find files
    files = find_png_files_in_dir(args.test_dir)

    # Create classifier
    clf = MyClassifier()

    # Let's go
    results = {}
    print("Processing files...")
    for file in files:
        print(file)
        file_full_path = os.path.join(args.test_dir, file)
        cls_id = -1
        try:
            cls_id = clf.classify_image(file_full_path)
            print(f"\t-> {cls_id}")
        except:
            print(f"Warning: failed to process file '{file_full_path}'.")
            raise
        results[file] = cls_id
    print("Done processing files.")

    # Save predictions
    save_classifications(results, args.output)
    print(f"Predictions saved in '{args.output}'.")

if __name__ == "__main__":
    main()

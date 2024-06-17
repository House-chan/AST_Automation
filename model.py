import cv2
import numpy as np
import pandas as pd
import psycopg2
import json
import os
from dotenv import load_dotenv

from os import path
from copy import copy
from scipy.ndimage import median_filter
from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from datetime import datetime
from scipy.stats import trim_mean


import PlateDetector
import MedicineDetector
from util import resize, moving_average, adjust_brightness

class PelletsDetector:
    def __init__(self):
        self.img_crop = None
        self.med_circles = None
        self.plate_circle = None
        self.plate_radius = None
        self.scale_factor = None
        self.med_loc = None
        self.med_rad = None
        self.inhibition_zone_diam = None
        self.pellets = None

    def process_image(self, image_path):
        if  image_path is not None:
            img = image_path
            # Convert to grayscale
            # img = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
          
            img = PlateDetector.resize(img, 2000)
            try:
                self.plate_circle = PlateDetector.detect(img)
                self.plate_radius = self.plate_circle[0, 0, 2]
            except Exception as e:
                print('PlateDetector Error',e)
                return str(e)  

            try:
                self.img_crop = PlateDetector.circle_crop(img, self.plate_circle, pad=0, normalize_size=True)
                self.med_circles = MedicineDetector.detect(self.img_crop, pad=0)
                plate_radius_real = 6.35 / 2
                self.scale_factor = plate_radius_real / np.mean(self.med_circles[0, :, 2])


                self.med_loc = [(float(self.med_circles[0, i, 0]), float(self.med_circles[0, i, 1])) for i in range(len(self.med_circles[0]))]
                self.med_rad = [int(np.floor(self.med_circles[0, i, -1])) for i in range(len(self.med_circles[0]))]
                self.pellets = [PlateDetector.circle_crop(self.img_crop, self.med_circles[0][i].reshape((1, 1, -1)), pad=150, normalize_size=False) for i in range(len(self.med_circles[0]))]
            except Exception as e:
                print('MedicineDetector Error',e)
                return str(e)  


            return self.img_crop
        else: return("image not found")
    
    def polar_coord_T(theta, r, cen_p:tuple)->tuple:
        """ transform polar coordinates to x ,y coordinates
            cen_p : a center point is the point(x,y) where r=0
        """
        
        # transfrom degree to radians.
        theta = 2*np.pi*theta/360
        
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        
        x = dx + cen_p[0]
        y = dy + cen_p[1]
        
        return np.array([x, y]).T
    
    def img_polar_transfrom(img, cen_p):
        """transform image to polar coordinates
        """
        img_size = img.shape[0]
        intencities = []

        for r in range(0,img_size):
            coor = PelletsDetector.polar_coord_T(np.arange(0,360, 1) , r, cen_p).astype(int)
            intencity = img[coor[:,1], coor[:,0]]
            intencities.append(intencity)

            # exit if cordinate out of image size
            if np.any(coor<=0) or np.any(coor>=img_size-1):
                break
            
        return np.array(intencities)
    
    def calculate_trimmed_mean(inten, proportion=0.1):
        """calculate trimmed mean of each ranged
        """
        return trim_mean(inten, proportion)
        
    def predict_diameter(self):
        # transform ALL to polar coordinates
        kernel = np.ones((4, 4), np.uint8)
        kernel2 = np.ones((6, 6), np.uint8)
        img_polar = []
        global closing_list
        closing_list = []

        for i in range(len(self.med_loc)):
            img_polar.append(PelletsDetector.img_polar_transfrom(self.img_crop, self.med_loc[i]))
            blurred_image = cv2.GaussianBlur(img_polar[i], (5, 5), 0)
            blurred_image = median_filter(blurred_image, size=5)
            blurred_image = adjust_brightness(blurred_image, 180)

            erosion = cv2.erode(blurred_image, kernel2, iterations=3)
            dilation = cv2.dilate(erosion, kernel2, iterations=3)
            closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel2)
            # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
            ## CV Normalize

            ##? Normal img
            normalized_closing = cv2.normalize(closing, None, 0, 255, cv2.NORM_MINMAX) 
            # closing_list.append(normalized_closing.astype(np.uint8))
            closing_list.append(normalized_closing.astype(np.uint8))

        intensity_r = []
        global_closing_list_sort = closing_list

        for i in range(len(self.med_loc)):
            for j in range(len(closing_list[i])):
                global_closing_list_sort[i][j] = np.sort(closing_list[i][j])

        intensity_r = []

        for i in range(len(self.med_loc)):
            intensity_r.append([PelletsDetector.calculate_trimmed_mean(inten) for inten in global_closing_list_sort()[i][0:440]])



        predict_radius_list = []
        change_points = []
        for i in range(len(self.med_loc)):
            x = np.arange(len(intensity_r[i]))
            y = np.array(intensity_r[i]).astype(float)
            y = y / 255 * 100
            y_mavg = moving_average(y, 2)
            # t_param = 1
            # Remove noise (medecien no need to calculate dy)
            for r in range(self.med_rad[0]):
                y_mavg[r] = 100
                
            dy = np.diff(y_mavg)

            dy[:self.med_rad[0] + 20] = np.abs(dy[:self.med_rad[0] + 20])


            y = y * 255 / 100
            
            # Set a threshold to determine significant change
            t_param = np.abs(np.mean(dy) - np.std(dy)) * 0.7
            threshold = np.mean(moving_average(abs(dy), 2)) / np.std(dy)  + 0.3
                
                # After 100 of range we dont use, As we know range can not be over 100 px
                # change_points = np.where((np.abs(dy) > threshold))[0]

                # Assuming dy and threshold are defined elsewhere
            change_points = np.arange(0, len(dy), 4)[(dy)[::4] > threshold]
            cp_status = True
            cp_count = 0
            fake_last_index = -1

        for (index_value, index)  in zip(change_points, range(len(change_points))):
            if (index == 0) : continue
            else : print(y[index_value] - y[index_value -1], index)
            previous_index_value = change_points[index - 1] if index > 0 else 0
            if (y[index_value] - y[index_value - 1] <= 0 and cp_status) : continue
            if ((y[index_value] - y[index_value - 1] <= 0) and not cp_status) : 
                print(index_value)
                fake_last_index = index -1
                break
            if (y[index_value] - y[index_value -1] > 0) : 
                cp_status = False
                cp_count += 1
            if cp_count >= 3: 
                if index_value - previous_index_value > 50:
                    fake_last_index = index -1
                    break           
                fake_last_index = index
                break

            if len(change_points) > 0:
                for index in change_points:
                    print(f"Index: {index}, Value: {y[index]}")
                predict_radius = x[change_points[fake_last_index]] - x[change_points[0]]
                predict_radius_list.append(predict_radius)
                
            else:
                print("Inhibition Zone Not Found")
                predict_radius_list.append(0)

        inhibition_zone_diam = []
        inhibition_zone_pixels = []
        for j in range(len(predict_radius_list)):
            inhibition_zone_diam.append(round((self.med_rad[j] + predict_radius_list[j])  * self.scale_factor * 2, 2))
            inhibition_zone_pixels.append(int(self.med_rad[j] + predict_radius_list[j]))
    
        self.inhibition_zone_pixels = inhibition_zone_pixels
        self.inhibition_zone_diam = inhibition_zone_diam

class Interpretator:
    def __init__(self):
        self.ast_id = None
        self.bacteria = None
        self.name = None
        self.new_data_point = None
        self.input_a = None
        self.input_b = None
        self.input_c = None


    def callable_zone(self):
        df = pd.read_csv(r"annotion.csv")
        s = df['S']
        i = df['I']
        r = df['R']
        sdd = df['SDD']

        input_a = self.input_a
        input_b = self.input_b
        input_c = self.input_c
        
        try:
            med_index = df.loc[(df['Antimicrobial Agent'] == input_b) & (df['Bacteria'] == input_c)].index.astype(int)[0]
        except IndexError:
            return f"No data for interpretion SIR {input_b}",'', input_a
        
        if pd.isna(s[med_index]) or pd.isna(i[med_index]) or pd.isna(r[med_index]):
            return "Data is NaN for the given index",'', input_a
        
        condition_s = s[med_index] # Extracting the condition from the list
        ass = i[med_index].split()
        condition_i = ass[0] + ' and input_a ' + ass[1]
        condition_r = r[1]

        if eval(str(input_a) + condition_s):
            print("Condition is S")
            return [input_b, 'S', input_a]
        elif eval(str(input_a) + condition_i):
            print("Condition is I")
            return [input_b, 'I', input_a]
        elif eval(str(input_a) + condition_r):
            print("Condition is R")
            return [input_b, 'R', input_a]
        else: return f"Antimicrobial or Bacteria not found {input_a}"
        

#Define Flask API

app = Flask(__name__)
#Make CORS Potocorrelation
CORS(app)
#Create Class Object for replete
pellets_detector = PelletsDetector()
interpretion = Interpretator()
load_dotenv()

# Retrieve database credentials from environment variables
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

conn = psycopg2.connect(
host=db_host,
database=db_name,
user=db_user,
password=db_password
)


@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return 'No file uploaded', 400
        
        image_file = request.files['image']
        
        # Check if the file is empty
        if image_file.filename == '':
            return 'No selected file', 400
        else:
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            try:
                image_array =  pellets_detector.process_image(np.array(image))
            except Exception as e:
                return 'Error while processing image', e
             
            # Convert the numpy array back to an image file (e.g., PNG)
            
            img_io = BytesIO()
            Image.fromarray(image_array).save(img_io, 'PNG')
            img_io.seek(0)
            pellets_detector.predict_diameter()
            # Send the image file
            return send_file(img_io, mimetype='image/png')
    except Exception as e:
        # Handle any exception that occurred during image processing
        return f"Error while import image {str(e)}", 500
        # return jsonify({'processed_image':  pellets_detector.process_image(image_path).tolist()}), 200


    
@app.route('/api/test_info', methods=['POST'])
def post_med_data():
    try:
        data = request.json

        # Check if all required fields are present
        required_fields = ['astId', 'bacteria', 'name', 'newDataPoint']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
            
        interpretion.ast_id = data['astId']
        interpretion.bacteria = data['bacteria']
        interpretion.name = data['name']
        new_data_points = data['newDataPoint']
        new_arrays = []

        print("data", data)
        print("interpretion.ast_id", interpretion.ast_id)
        print("interpretion.bacteria", interpretion.bacteria)
        print("interpretion.name", interpretion.name)
        print("new_data_points", new_data_points)

        try:
            print(len(new_data_points))
            for i in range(len(new_data_points)):

                interpretion.input_a = float(round(new_data_points[i][1] * pellets_detector.scale_factor * 2, 2)) 
                interpretion.input_b = str(new_data_points[i][0])
                interpretion.input_c = str(interpretion.bacteria)
                print("input_a", interpretion.input_a)
                print("input_b", interpretion.input_b)
                print("input_c", interpretion.input_c)
                new_arrays.append(interpretion.callable_zone())
        except Exception as e:
            return e, "Error when interpretion inhibition zone"
        # Return a response
        return new_arrays, 200
    
    except Exception as e:
        # If an error occurs during processing, return an error response
        return jsonify({"error": str(e)}), 500
    


@app.route('/api/get_all_gallery', methods=['GET'])
def get_all_gallery():
    try:
        with conn.cursor() as cur:  # Establish database connection 
            cur.execute('SELECT * FROM public."SummaryResult" ORDER BY "astID" ASC')
            results = cur.fetchall()
            # Format results as a list of dictionaries (assuming column names)
            data = [dict(zip([column[0] for column in cur.description], row)) for row in results]
            return jsonify(data)  # Return data as JSON
        
    except (Exception, psycopg2.Error) as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500  # Return error on failure
  
import base64
@app.route('/api/med_info', methods=['GET'])
def get_med_info():
    if pellets_detector.med_loc is not None and pellets_detector.med_rad is not None:
        # med_json = [{'x': loc[0], 'y': loc[1], 'radius': rad, 'predict_diameter': diam} for loc, rad, diam in zip(pellets_detector.med_loc, pellets_detector.med_rad, pellets_detector.inhibition_zone_diam)]
        
        images_data = []
        for img_array in pellets_detector.pellets:
            # Convert the NumPy array to an image
            img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            # Convert the image to a byte array
            _, img_encoded = cv2.imencode('.png', img)
            # Encode the byte array as a base64 string
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            # Add the base64 string to the list
            images_data.append(img_base64)
        med_data = [(loc[0], loc[1], diam, images_data) for loc, diam, images_data in zip(pellets_detector.med_loc, pellets_detector.inhibition_zone_pixels, images_data)]
        print(med_data)
        return (med_data), 200
    else:
        return jsonify({'error': 'Medicine information not available. Please Process an image first.'}), 404
    
@app.route('/api/add_data', methods=['POST'])
def add_data():
    try:
        new_data = request.get_json()

        # Extract values from the input JSON
        ast_id = new_data['astId']
        bacteria_name = new_data['bacteria']
        user_name = new_data['name']

        logs = []
        for item in new_data['newDataPoint']:
            antibiotic_name, sir, inhibition_diam = item[0], item[1], item[2]
            logs.append({
                'antibioticsName': antibiotic_name,
                'SIR': sir,
                'inhibitionDiam': inhibition_diam
            })

        # Create the final output JSON
        output_json = {
            'astID': ast_id,
            'bacteriasName': bacteria_name,
            'userName': user_name,
            'logs': logs
        }

        new_data = output_json
        cur = conn.cursor()

        with conn.cursor() as cur:
            # Insert into SummaryResult
            cur.execute(
                'INSERT INTO public."SummaryResult"("astID", "addedAt", "bacteriasName", "userName")'
                'VALUES (%s, CURRENT_TIMESTAMP, %s, %s) RETURNING "astID"', 
                (new_data['astID'], new_data['bacteriasName'], new_data['userName'])
                )

            for item in new_data['logs']:  # Assuming the log data is in an array called 'logs'
                cur.execute(
                    'INSERT INTO public."Log"( "antibioticsName", "SIR", "inhibitionDiam", "astID") '
                    'VALUES (%s, %s, %s, %s)',
                    (item['antibioticsName'], item['SIR'], item['inhibitionDiam'], new_data['astID'])
                )

            conn.commit()
            return jsonify({"message": "Data added successfully to both tables"})
        
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        print(e)
        return (f"error{e}", 500)

@app.route('/api/get_data_by_astID', methods=['POST'])
def get_data_by_astID():
    data = request.get_json()
    astID = data['astID']

    try:
        with conn.cursor() as cur:
            # Fetch data from SummaryResult
            cur.execute(
                """
                SELECT * 
                FROM public."SummaryResult" 
                WHERE "astID" = %s 
                """, 
                (astID,)
            )
            summary_result_data = cur.fetchone()

            if not summary_result_data:
                return jsonify({"error": "astID not found"}), 404  # Error handling

            # Fetch logs from Log table
            cur.execute(
                """
                SELECT * 
                FROM public."Log" 
                WHERE "astID" = %s 
                """, 
                (astID,)
            )

            log_data = cur.fetchall()

            # Format the logs data
            # Combine the results into a single response
            print(log_data)
            response_data = {
                "astID": summary_result_data[0],
                "date": summary_result_data[1].strftime("%d %B %Y"),  # Format date
                "bacteriasName": summary_result_data[2],
                "userName": summary_result_data[3],
                "logs": [
                    [
                        log_entry[3],  # antibioticsName
                        log_entry[1],  # SIR
                        log_entry[2],  # inhibitionDiam
                        # Leave out "astID" since it's redundant
                    ] for log_entry in log_data
                ]
            }

            # Return the formatted JSON response
            return jsonify(response_data)
    except (Exception, psycopg2.Error) as e:
        conn.rollback()
        return jsonify({"error": "Database error occurred"}), 500  # Internal Server Error


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)

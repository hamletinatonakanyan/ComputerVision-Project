
 from zipfile import ZipFile
import os
import glob
import shutil
import random
import re
import PIL
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%% md

# Load data

#%%

# unzip data in the project with creating new input_data folder

zipped_path = 'E:\PythonMentor\computer_vision\data/500MB.zip'
with ZipFile(zipped_path, 'r') as zipped:
    zipped.extractall(path='input_data')


#%%

# read  csv file (information about cars) through pandas
df_car = pd.read_csv('input_data/500MB/labels.csv')

#%% md

# Data Exploration

#%%

# look at the features and values
df_car.head()

#%%

# look at the information about data
df_car.info()

#%%

# check if there is not null values
df_car.isnull().sum()

#%% md

# choose the feature for further modeling
# Plotting features' value counts

#%%

# look at the Color feature
df_car.Color.value_counts().plot(kind='bar')
plt.title('Amount of cars with specific color')
plt.show()

print(df_car.Color.value_counts())

#%%

# look at the Style feature

plt.figure(figsize=(9, 6))
sns.set_style('darkgrid')
sns.countplot(x=df_car.Style)
plt.xticks(rotation=65)
plt.title('Amount of cars with each style')
plt.show()

print(df_car.Style.value_counts())

#%%

# look at the type feature

plt.figure(figsize=(9, 6))
sns.set_style('darkgrid')
sns.countplot(x=df_car.year)
plt.xticks(rotation=90)
plt.title('Amount of cars produced in different years')
plt.show()

print(df_car.year.value_counts())

#%%

# look at the brand feature

plt.figure(figsize=(9, 6))
sns.set_style('darkgrid')
sns.countplot(x=df_car.brand)
plt.xticks(rotation=65)
plt.title('Amount of cars of each brand')
plt.show()

print(df_car.brand.value_counts())

#%% md

# Values distributions
# Look at the values distributions through box plot

#%%

plt.figure(figsize=(9, 6))
df_car.Color.value_counts().plot.box()
plt.title('Car color values box plot')
plt.show()


#%%

plt.figure(figsize=(9, 6))
df_car.Style.value_counts().plot.box()
plt.title('Car style values box plot')
plt.show()


#%%

plt.figure(figsize=(9, 6))
df_car.year.value_counts().plot.box()
plt.title('Car year values box plot')
plt.show()

#%%

plt.figure(figsize=(9, 6))
df_car.brand.value_counts().plot.box()
plt.title('Car brand values box plot')
plt.show()


#%%

# look at the model values distribution through histogram
df_car.brand.value_counts().hist()
plt.title('Car brand values distribution')
plt.show()

#%% md

# Decision
# As we see from above data and plots exploring good distributions have 'year' and 'brand' features.


#%%
# function for checking if images are corrupted or not

def is_corrupt(img):

    """Returns None if the file is okay,
    returns message with filename if the file is corrupt."""

    try:
        im = PIL.Image.open(img)
        im.verify()
    except (IOError, SyntaxError) as e:
        return str(e)
    return None

#%%
# function for storing corrupted images from all folders


def get_corrupted_images(dir_path, extension):

    """Returns paths of all corrupted images."""

    corrupted_img = []

    for img_path in dir_path:
        if img_path.endswith(extension):
            status = is_corrupt(img_path)
            if status is not None:
                corrupted_img.append(img_path)

    return corrupted_img


#%%
# store and show all corrupted images paths

images_path = 'input_data/500MB/images'
id_folders_path = glob.glob(f"{images_path}/*/*.jpg")

corrupted_images = get_corrupted_images(id_folders_path, '.jpg')
print(corrupted_images)


#%%
# function for creating subfolders for train and test data

def create_subfolders(folder, all_features):
    for feature in all_features:
        if not os.path.isdir(f'{folder}/train/{feature}') or not os.path.isdir(f'{folder}/test/{feature}'):
            os.makedirs(f'{folder}/train/{feature}')
            os.makedirs(f'{folder}/test/{feature}')


#%% md

# Working with brand feature
# Create structure, in order to load the data and feed it to the further model


#%%
# check unique brands and store in the array

all_brands = df_car.brand.unique()
print(f'Amount of brands is |{len(all_brands)}| and the names are:\n{all_brands}')

#%%
# create data folder with 2 test and train subfolders\
# in each one separate folders with name of brands for\
# further storing of car images by brands.

fold = 'brand_data'
create_subfolders(fold, all_brands)

#%%
""" Chek non corrupted images and copy the images of each brand
to the appropriate subfolders of train folder. """

# set regex pattern  for finding folder id in the folder path
id_pattern = r'\d*$'
count = 0

for path in id_folders_path:

    re_path = os.path.split(path)[0]
    folder_id = int(re.findall(id_pattern, re_path)[0])

    # in the car dataframe find the brand with given id
    df_index = df_car[df_car.id==folder_id].index[0]
    car_brand = df_car.iloc[df_index].brand

    # chek non corrupted images and copy the images of each brand to the appropriate subfolders of train folder
    dest_path = f'brand_data/train/{car_brand}/{car_brand}{count}.jpg'
    if path not in corrupted_images:
        shutil.copy(path, dest_path)

    count += 1


#%%
""" Rename images, adding at the end of image's path the ordered
number of image from 0 to the amount of images of that brand. """

for brand_path in glob.glob('brand_data/train/*'):
    # find only brand name in the subfolders path
    brand_name = os.path.split(brand_path)[1]

    # set img_count variable to count number of images in exact subfolder with exact brand name
    # rename images, adding at the end of image path the ordered number of image from 0 to amount of images of that brand
    img_count = 0
    for img_path in glob.glob(f'{brand_path}/*.jpg'):

        dest_name_for_rename = f'{brand_path}/{brand_name}{img_count}.jpg'
        if not os.path.exists(dest_name_for_rename):
            os.rename(img_path, dest_name_for_rename)

        img_count += 1


#%%
""" Move the 10% images of each brand from the train folder
to the appropriate subfolders of test folder. """

for brand_path in glob.glob('brand_data/train/*'):
    # match only brand name in the subfolders path with brand names
    brand_name = os.path.split(brand_path)[1]

    # get all paths of images of exact brand folder and store in the list
    # set test_size variable- amount(10% of train size) of images that will be moved to the test folder
    files_list = glob.glob(f'{brand_path}/*.jpg')
    test_size = (len(files_list) * 10) // 100

    # randomly chose images from train set to move them to test set
    file_pats_for_test = random.sample(files_list, k=test_size)

    # set count variable to count number of images in exact subfolder with exact brand name in the test folder
    # move 10% of images of each brand subfolder in the train folder to the appropriate subfolders in the test folder
    test_brand_img_count = 100
    for img_path in file_pats_for_test:
        test_dest_path = f'brand_data/test/{brand_name}/{brand_name}{test_brand_img_count}.jpg'

        if not os.path.exists(test_dest_path):
            shutil.move(img_path, test_dest_path)

        test_brand_img_count += 1


#%% md

# Working with "year" feature

#%%
# check unique years and store in the array

all_years = df_car.year.unique()
print(f'Amount of years is |{len(all_years)}| and the names are:\n{all_years}')


#%%

# create data folder with 2 test and train subfolders\
# in each one separate folders with name of years for\
# further storing car images by years.

year_folder = 'year_data'
create_subfolders(year_folder, all_years)


#%%

""" Chek non corrupted images and copy the images of each year
to the appropriate subfolders of train folder. """

id_pattern = r'\d*$'
count = 0
for path in id_folders_path:
    re_path = os.path.split(path)[0]
    folder_id = int(re.findall(id_pattern, re_path)[0])

    # in the car dataframe find the year with given id
    df_index = df_car[df_car.id == folder_id].index[0]
    car_year = df_car.iloc[df_index].year

    # chek non corrupted images and copy the images of each year to the appropriate subfolders of train folder
    dest_path = f'year_data/train/{car_year}/{car_year}_{count}.jpg'
    if path not in corrupted_images:
        shutil.copy(path, dest_path)
    count += 1


#%%
""" Move the 10% images of each year from the train folder
to the appropriate subfolders of test folder. """

for year_path in glob.glob('year_data/train/*'):
    # match only year name in the subfolders path with year names
    year_name = os.path.split(year_path)[1]

    # get all paths of images of exact year folder and store in the list
    # set test_size variable- amount(10% of train size) of images that will be moved to the test folder
    files_list = glob.glob(f'{year_path}/*.jpg')

    test_size = (len(files_list) * 10) // 100
    if len(files_list) !=0 and len(files_list) < 10:
        test_size = 2

    # randomly chose images from train set to move them to test set
    file_pats_for_test = random.sample(files_list, k=test_size)

    # set count variable to count number of images in exact subfolder with exact year name in the test folder
    # move 10% of images of each year subfolder in the train folder to the appropriate subfolders in the test folder
    test_year_img_count = 1
    for img_path in file_pats_for_test:
        test_dest_path = f'year_data/test/{year_name}/{year_name}_{test_year_img_count}.jpg'

        if not os.path.exists(test_dest_path):
            shutil.move(img_path, test_dest_path)

        test_year_img_count += 1


#%% md

# 'color'  feature
# Make folders for 'color' feature.
# Chose small amount of images in every folder for making data normal distributed

#%%
# check unique years and store in the array

all_colors = df_car.Color.unique()
print(f'Amount of years is |{len(all_colors)}| and the names are:\n{all_colors}')


#%%
# create data folder with 2 test and train subfolders\
# in each one separate folders with name of colors for\
# further storing car images by colors.

color_folder = 'color_data'
create_subfolders(color_folder, all_colors)


#%%
""" Chek non corrupted images and copy 10 images of each color
to the appropriate subfolders of train folder. """

id_pattern = r'\d*$'
count = 0

for path in id_folders_path:
    re_path = os.path.split(path)[0]
    folder_id = int(re.findall(id_pattern, re_path)[0])

    # in the car dataframe find the color with given id
    df_index = df_car[df_car.id == folder_id].index[0]
    car_color = df_car.iloc[df_index].Color

    # chek non corrupted images and copy the images of each color to the appropriate subfolders of train folder
    dest_path = f'color_data/train/{car_color}/{car_color}{count}.jpg'
    if path not in corrupted_images:
        shutil.copy(path, dest_path)
    count += 1

#%%

for color_path in glob.glob('color_data/train/*'):
    # match only color name in the subfolders path with color names
    color_name = os.path.split(color_path)[1]

    # get all paths of images of exact color folder and store in the list
    files_list = glob.glob(f'{color_path}/*.jpg')

    # in each subfolder keep only les than 10 images
    if len(files_list) >= 10:
        for file in files_list[10:]:
            os.unlink(file)


#%%

""" Move the 1 image of each color from the train folder
to the appropriate subfolders of the test folder. """

for color_path in glob.glob('color_data/train/*'):
    # match only year name in the subfolders path with year names
    color_name = os.path.split(color_path)[1]

    # get all paths of images of exact year folder and store in the list
    files_list = glob.glob(f'{color_path}/*.jpg')

    # randomly chose 1 image from train set to move it to the test set
    file_pats_for_test = random.sample(files_list, k=2)

    count = 1
    for img_path in file_pats_for_test:
        test_dest_path = f'color_data/test/{color_name}/{color_name}_{count}.jpg'

        if not os.path.exists(test_dest_path):
            shutil.move(img_path, test_dest_path)

        count += 1

#%%

#for un_path in glob.glob('color_data/train/*/*.jpg'):
    #os.unlink(un_path)


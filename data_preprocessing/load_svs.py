
#provide all imports here 
from openslide import (OpenSlide, OpenSlideError,
        OpenSlideUnsupportedFormatError)
import re
import sys
import PIL
import numpy as np
import os
from PIL import Image, ImageDraw
from openslide.deepzoom import DeepZoomGenerator as dz
import cv2
import math
import pandas as pd

#Import xml dependencies
from xml.dom import minidom
import xml.etree.ElementTree as ET


'''
 The function parse takes an xml file as input (with the region and vertices coordinates of that input) for a sample whole slide images.
 The fuction parses through the coordinates of every region and returns and array of these coordinates. 
'''
def parse(inputfile,regions, vertices):
    """
    Input: XML file to get coordinates from, the region of interest and the vertices of that region
    Output: An array of connected points for each region [Y,X,Z] 
    """
    
    tree = ET.parse(inputfile)
    root = tree.getroot()
	
    #dictionary of point coordinates
    images = {}
    for region in root.iter(regions):
        coordinates = []
        for vertex in region.iter(vertices):
            temp = [int(i) for i in vertex.attrib.values()]
            coordinates.append(temp)
        
        #append the coordinates to images dictionary
        images.update({region.attrib["Id"]: np.array(coordinates)})
    
    return images


'''Get_bounds function takes the patches and slide as input, extract the region of interest, then save it to the destination directory'''
def get_bounds(patches, slide_location):
      
    """
    Input: Image patches as dictionary of n keys with matrix of (Y,X,Z) Coordinates per patch
    Output: an image mask for each valid bounding box 
    """
    slide = OpenSlide(slide_location)
    print(slide.level_dimensions)
    slide_size = slide.level_dimensions[0] 
    #mask_zero = np.zeros(shape = slide_size, dtype=float)
    boxes = []
    masks = []
    values = patches.values()
    i = 0
    
    mask =PIL.Image.new('L', slide_size, 0)
    good_region_mask= PIL.Image.new('L', slide_size, 1)
    
    for patch in values:
        first_coord = patch[0]
        last_coord = patch[len(patch)-1]
        
        area = abs(first_coord[0]-last_coord[0])* abs(first_coord[1]-last_coord[1])
        
        
            
        if area < 40000:  # annotation is circle
              
            y_max = max(patch[:,0])
            y_min = min(patch[:,0])
            
            x_max = max(patch[:,1])
            x_min = min(patch[:,1])
            
            #make sure annotations are in slide scope
           # if max(patch[:,1])< slide_size[0]:
            #draw the polygon on the mask and fill with ones
           
           # print ((patch[:,0]))
           # print((patch[:,1]))
            ImageDraw.Draw(mask).polygon(zip(patch[:,1],patch[:,0]), outline=1, fill=1)
            
            #store the result in mask_result for further computation
            mask_result = np.array(mask)
            
            #change the outline and fill to white for further use
            ImageDraw.Draw(good_region_mask).polygon(zip(patch[:,1],patch[:,0]), outline=0, fill=0)
                    
            
            #resize  the mask to the region of interest dimensions
            
            mask_result = mask_result[y_min:y_max,x_min:x_max]
            #mask_result = np.expand_dims(mask_result, axis = 3)
            mask_result = cv2.cvtColor(mask_result, cv2.COLOR_GRAY2RGB)
            
            #mask_result = mask_result
            
            
            
            this_region = slide.read_region((x_min,y_min),0,(x_max-x_min , y_max-y_min)).convert('RGB')
            this_region.save("o"+str(i)+".png")
            this_region = cv2.imread("o"+str(i)+".png")#np.array(this_region)
       
           #     print("Error ============> Annotations out of bounds for slide"+str(slide_location)+str(max(patch[:,0]))+"slide size==>"+str(slide_size[1]))
            
            print ("Original croped mask shape {}".format(mask_result.shape))
            print ("Original croped region shape {}".format(this_region.shape))
            
            if mask_result.shape == this_region.shape:
                result = np.multiply(mask_result,this_region)
                boxes.append(result)
                masks.append(mask_result)
                
            else:
                print("slide "+str(i)+" shape is not equal")
            i = i+1 
        else:
            #change the outline and fill to white for further use
            ImageDraw.Draw(good_region_mask).rectangle([(min(patch[:,1])-100,min(patch[:,0])-1000),(max(patch[:,1])+1000,max(patch[:,0])+1000)], outline=0, fill=0)
                    
        
    return boxes, good_region_mask, slide, values 

def get_good_ccnuclei(mask, slide,values):
    """
    Input: mask  -- a resulting mask containing just good nuclei, and all the others regions are white
           slide -- complete region of interest from the original slide
           values -- array of point annotations for each patch

    output: return an array of resultant matrices for each slide and its corresponding mask
    """
    globalX_min = min(values[0][:,1])
    globalX_max = max(values[0][:,1])
    for patch in values:
        #get minimum x and x max
        x_min = min(patch[:,1])
        x_max = max(patch[:,1])
        if x_min < globalX_min:
            global_min = x_min
        if x_max > globalX_max:
            globalX_max = x_max

    x_start = int(globalX_min-1000)
    width = (globalX_max-globalX_min)+1000
    height = slide.level_dimensions[0][1]
    x_end =  int(x_start+width)
    
    if globalX_max > math.floor(slide.level_dimensions[0][0]/2):
        mask = np.array(mask)
        mask = mask[:,x_start:x_end]


    #chunk_width = int(math.floor(width/2))
    #chunk_height = int(math.floor(height/2))
        ymiddle = int(math.floor(slide.level_dimensions[0][1]/2))
        #stratify each region of the mask into four equal part
        mask1 = mask[:ymiddle,x_start:x_end]
        mask1 = cv2.cvtColor(mask1, cv2.COLOR_RGB2HSV)
        mask2 = mask[ymiddle:slide.level_dimensions[0][1],x_start:x_end]
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_RGB2HSV)
    #mask3 = mask[chunk_height:(chunk_height+chunk_height),:chunk_width]
    #mask3 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)
    #mask4 = mask[chunk_height:(chunk_height+chunk_height), chunk_width:(chunk_width+chunk_width)]
    #mask4 = cv2.cvtColor(mask4, cv2.COLOR_GRAY2BGR)


    #stratify each region of the slide into four equal parts
    #print(slide.level_dimensions[0][0])
    #print(slislide.level_dimensions[0][1])
        ystart = 0
        print("width "+ str(x_end-x_start))
        print("height "+ str(ymiddle))
        slide1 = slide.read_region((0,0),0,((x_end-x_start),ymiddle)).convert('RGB')
        slide1.save("slide_roi1.png")
        slide1 = cv2.imread("slide_roi1.png")

        slide2 = slide.read_region((0,smiddle),0,((x_end-x_start), slide.level_dimensions[0][1])).convert('RGB')
        slide2.save("slide_roi2.png")
        slide2 = cv2.imread("slide_roi2.png")

    #slide3 = slide.read_region((x_start,chunk_height),0,(chunk_width , chunk_height)).convert('LA')
    #slide3.save("slide_roi3.png")
    #slide3 = cv2.imread("slide_roi3.png")

    #slide4 = slide.read_region(((x_start+chunk_width),chunk_height),0,(chunk_width , chunk_height)).convert('LA')
    #slide4.save("slide_roi4.png")
    #slide4 = cv2.imread("slide_roi4.png")
        print(mask1.shape )
        print(slide1.shape)
        result1 = np.multiply(mask1,slide1)
        result2 = np.multiply(mask2,slide2)
    #result3 = np.multiply(mask3,slide3)
    #result4 = np.multiply(mask,slide1)

    return [result1, result2]                       


def get_good_nuclei(mask, slide,values):
    """
    Input: mask  -- a resulting mask containing just good nuclei, and all the others regions are white
           slide -- complete region of interest from the original slide 
           values -- array of point annotations for each patch 
    
    output: return an array of resultant matrices for each slide and its corresponding mask
    """
    globalX_min = min(values[0][:,1])
    globalX_max = max(values[0][:,1])
    for patch in values:
        #get minimum x and x max
        x_min = min(patch[:,1])
        x_max = max(patch[:,1])
        if x_min < globalX_min:
            global_min = x_min
        if x_max > globalX_max:
            globalX_max = x_max
    
    x_start = int(globalX_min-1000)
    
    width = (globalX_max-globalX_min)+1000
    height = slide.level_dimensions[0][1]
    x_end =  x_start+width
    
    mask = np.array(mask)
    mask = mask[:,x_start:x_end]
    
    
    chunk_width = int(math.floor(width/2))
    chunk_height = int(math.floor(height/2))
    #stratify each region of the mask into four equal part
    mask1 = mask[:chunk_height,:chunk_width]
    mask1 = cv2.cvtColor(mask1, cv2.COLOR_RGB2HSV)
    mask2 = mask[ :chunk_height,chunk_width:(chunk_width+chunk_width)]
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_RGB2HSV)
    mask3 = mask[chunk_height:(chunk_height+chunk_height),:chunk_width]
    mask3 = cv2.cvtColor(mask3, cv2.COLOR_RGB2HSV)
    mask4 = mask[chunk_height:(chunk_height+chunk_height), chunk_width:(chunk_width+chunk_width)]
    mask4 = cv2.cvtColor(mask4, cv2.COLOR_RGB2HSV)
    
    
    #stratify each region of the slide into four equal parts
    slide1 = slide.read_region((x_start,0),0,(chunk_width , chunk_height)).convert('RGB')
    slide1.save("slide_roi1.png")
    slide1 = cv2.imread("slide_roi1.png")
    
    slide2 = slide.read_region(((x_start+chunk_width),0),0,(chunk_width , chunk_height)).convert('RGB')
    slide2.save("slide_roi2.png")
    slide2 = cv2.imread("slide_roi2.png")
    
    slide3 = slide.read_region((x_start,chunk_height),0,(chunk_width , chunk_height)).convert('RGB')
    slide3.save("slide_roi3.png")
    slide3 = cv2.imread("slide_roi3.png")
    
    slide4 = slide.read_region(((x_start+chunk_width),chunk_height),0,(chunk_width , chunk_height)).convert('RGB')
    slide4.save("slide_roi4.png")
    slide4 = cv2.imread("slide_roi4.png")
    
    result1 = np.multiply(mask1,slide1)
    result2 = np.multiply(mask2,slide2)
    result3 = np.multiply(mask3,slide3)
    result4 = np.multiply(mask4,slide4)
    
    return [result1,result2, result3, result4]
    
    
# Threshold the input images to isolate each nuclei
# #To Do
# 
# for each resulting box, create a window of 50 by 50 and stride across the box, while keeping the results of those with threshold of white pixels(255) less than 50 % and discarding the rest
# 


# 512x512
def get_patches(img, dest_dir, slide_counter, patch_counter):
   """
   Inputs:
          Img -- region of interest to be croped in samples of 50*50 based on threshold
          dest_dir -- Destination directory of training images
   """

   img_shape = img.shape 
   tile_size = (256, 256) 
   offset = (100, 100)
   counter = 0
   #df = pd.DataFrame(columns=["I","J","Counter","Image"])
   results = pd.read_csv("./heat_map_scores.csv")
    #for j in range(len(y_classes)):
    #    df.loc[j] = [ground_truth[j],y_classes[j]]

    #df.to_csv("./md_plosone_heresults/"+str(i)+"md_HEresults.csv")
    #print("Saved to: "+str(i)+"md_HEresults.csv")

   #collect positive cancer cells
   for i in xrange(int(math.ceil(img_shape[0]/(offset[1]*1.0)))):
       for j in xrange(int(math.ceil(img_shape[1]/(offset[0]*1.0)))): 
           cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1],img_shape[0]),offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]

           #treshold for the pixel values greater than 50 and less than 180  then eliminate white and black patches
           if (cropped_img.mean() > 50) and (cropped_img.mean() < 180) and (len(list(set(cropped_img.flatten())))>50):
           
               #print(len(list(set(cropped_img.flatten()))))
                    
               # Debugging the tiles
               cv2.imwrite(dest_dir+"/"+ str(i) + "_" + str(j) +"debug"+str(counter)+ ".png", cropped_img)
               #df.loc[counter] = [i, j, counter, cropped_img]
               #cropped_img.fill(results.score[counter])
               img[offset[1]*i:min(offset[1]*i+tile_size[1],img_shape[0]),offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])] = results.score[counter]*155
               counter = counter + 1
   #df.to_csv("./heat_map.csv")
   cv2.imwrite("heat.png", img)    

'''
This function rearranges the WSI in source folder with the corresponding XML containing the annotations
'''
def rearrangeDir():
	count_svs = 0
	count_xml = 0
	svs = []
	xml = []
	for filename in os.listdir("./poorly_differentiated/CK/valid"):
		if filename.endswith(".svs"):
			svs.append(filename)
		
		if filename.endswith(".xml"):
			xml.append(filename)

	svs = sorted(svs)
	xml = sorted(xml)
	
	return [svs, xml] 
	



##Sample test

images = parse("./1003965.xml",regions = 'Region', vertices = 'Vertex')
    
positive_nuclei, good_mask, slide, values = get_bounds(images, "./1003965.svs")
    
negative_nuclei_roi = get_good_nuclei(mask = good_mask, slide = slide, values = values)
    
slide_counter = 0    
    #print("Here is share of slide with good nuclei" + str(negative_nuclei_roi.shape))
patch_counter = 0 
for positive in positive_nuclei:
    get_patches(positive, "./heat_map", slide_counter, patch_counter)
    patch_counter += 1
    
patch_counter = 0
for negative in negative_nuclei_roi:
get_patches(negative, "./pytorch/largeCK_data/train/1", slide_counter, patch_counter)
    patch_counter += 1
    
    
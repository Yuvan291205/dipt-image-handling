#!/usr/bin/env python
# coding: utf-8

# In[31]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[32]:


img = cv2.imread('rameshwaram.jpg', cv2.IMREAD_COLOR)


# In[33]:


img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


# In[34]:


print(img_rgb.shape)


# In[35]:


plt.imshow(img_rgb)
plt.show()


# In[39]:


img_color = cv2.imread('rameshwaram.jpeg')
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)


# In[40]:


plt.imshow(img_rgb)
plt.show()
print(img_rgb.shape)


# In[41]:


height, width, _ = img_rgb.shape
crop = img_rgb[50:height-50, 50:width-50]  # Crop edges, keeping the road centered
plt.imshow(crop)
plt.title("Cropped Region (Road with Edges Removed)")
plt.axis("off")
plt.show()
print(crop.shape)


# In[42]:


res = cv2.resize(crop, None, fx=2, fy=2)


# In[43]:


flip = cv2.flip(res, 1)
plt.imshow(flip)
plt.title("Flipped Horizontally")
plt.axis("off")
plt.show()


# In[45]:


img = cv2.imread('rameshwaram.jpeg', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_rgb.shape)


# In[46]:


text_img = img_rgb.copy()
cv2.putText(text_img, "Rameswaram Bridge, August 20, 2025", (200, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
plt.imshow(text_img)
plt.title("New image")
plt.show()


# In[49]:


height, width, _ = img_rgb.shape
rcol = (255, 0, 255)  # Magenta color
cv2.rectangle(text_img, (0, 0), (width-1, height-1), rcol, 3)
plt.title("Annotated image")
plt.imshow(text_img)
plt.show()


# In[51]:


img = cv2.imread('rameshwaram.jpeg', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[52]:


m = np.ones(img_rgb.shape, dtype="uint8") * 50


# In[53]:


img_brighter = cv2.add(img_rgb, m)
img_darker = cv2.subtract(img_rgb, m)


# In[54]:


plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("Original Image"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(img_brighter); plt.title("Brighter Image"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(img_darker); plt.title("Darker Image"); plt.axis("off")
plt.show()


# In[55]:


matrix1 = np.ones(img_rgb.shape, dtype="float32") * 1.1
matrix2 = np.ones(img_rgb.shape, dtype="float32") * 1.2
img_higher1 = cv2.multiply(img_rgb.astype("float32"), matrix1).clip(0, 255).astype("uint8")
img_higher2 = cv2.multiply(img_rgb.astype("float32"), matrix2).clip(0, 255).astype("uint8")


# In[57]:


matrix_lower = np.ones(img_rgb.shape, dtype="float32") * 0.8
img_lower = cv2.multiply(img_rgb.astype("float32"), matrix_lower).clip(0, 255).astype("uint8")


# In[59]:


plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("Original Image"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(img_lower); plt.title("Lower Contrast (0.8x)"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(img_higher1); plt.title("Higher Contrast (1.4x)"); plt.axis("off")
plt.show()


# In[60]:


b, g, r = cv2.split(img_rgb)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.imshow(b, cmap='gray'); plt.title("Blue Channel"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(g, cmap='gray'); plt.title("Green Channel"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(r, cmap='gray'); plt.title("Red Channel"); plt.axis("off")
plt.show()


# In[61]:


merged_rgb = cv2.merge([r, g, b])
plt.figure(figsize=(5, 5))
plt.imshow(merged_rgb)
plt.title("Merged RGB Image")
plt.axis("off")
plt.show()


# In[62]:


hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1); plt.imshow(h, cmap='gray'); plt.title("Hue Channel"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(s, cmap='gray'); plt.title("Saturation Channel"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(v, cmap='gray'); plt.title("Value Channel"); plt.axis("off")
plt.show()


# In[63]:


merged_hsv = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
combined = np.concatenate((img_rgb, merged_hsv), axis=1)
plt.figure(figsize=(10, 5))
plt.imshow(combined)
plt.title("Original Image & Merged HSV Image")
plt.axis("off")
plt.show()


# In[64]:


cropped_img = crop  
matrix1 = np.ones(cropped_img.shape) * 0.8
matrix2 = np.ones(cropped_img.shape) * 1.2
img_lower = np.uint8(cv2.multiply(np.float64(cropped_img), matrix1))
img_higher = np.uint8(cv2.multiply(np.float64(cropped_img), matrix2))
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_lower[:,:,::-1]); plt.title('Lower Contrast')
plt.subplot(132); plt.imshow(cropped_img[:,:,::-1]); plt.title('Original')
plt.subplot(133); plt.imshow(img_higher[:,:,::-1]); plt.title('Higher Contrast')
plt.show()


# In[ ]:





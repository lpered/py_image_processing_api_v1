from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from PIL import Image, ImageFilter, ImageOps
import httpx
from io import BytesIO
import cv2
import numpy as np
import boto3
import time
import os
from dotenv import load_dotenv  # Import dotenv to load environment variables

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Retrieve bucket configuration from environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKET_ENDPOINT = os.getenv("BUCKET_ENDPOINT")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=f"https://{BUCKET_ENDPOINT}",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

class ImageProcessingRequest(BaseModel):
    image_url: str
    operation: str  # e.g., "resize", "grayscale", "remove_background", "thumbnail"
    width: Optional[int] = None  # For resizing
    height: Optional[int] = None  # For resizing

async def download_image(image_url: str) -> Image.Image:
    """Asynchronously download an image from a URL and return a PIL Image."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

def resize_image(image: Image.Image, width: Optional[int], height: Optional[int]) -> Image.Image:
    """Resize the image while maintaining aspect ratio if only one dimension is provided."""
    original_width, original_height = image.size
    if width and not height:
        aspect_ratio = original_height / original_width
        height = int(width * aspect_ratio)
    elif height and not width:
        aspect_ratio = original_width / original_height
        width = int(height * aspect_ratio)
    elif not width or not height:
        raise HTTPException(status_code=400, detail="Either width or height must be provided for resizing.")
    return image.resize((width, height))

def remove_background(image: Image.Image) -> Image.Image:
    """Remove the background from the image and make it transparent using OpenCV."""
    # Convert the PIL image to an OpenCV image
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to create a binary mask
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask with the largest contour (assumes the object is the largest)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Create a 4-channel (RGBA) image with a transparent background
    b, g, r = cv2.split(open_cv_image)
    alpha = mask  # Use the mask as the alpha channel
    rgba_image = cv2.merge((b, g, r, alpha))

    # Convert the result back to a PIL image
    return Image.fromarray(rgba_image, mode="RGBA")

def upload_to_bucket(image: BytesIO, filename: str) -> str:
    """Upload the image to the bucket and return the URL."""
    try:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=filename, Body=image, ContentType="image/png")
        return f"https://{BUCKET_ENDPOINT}/{BUCKET_NAME}/{filename}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image to bucket: {str(e)}")

@app.post("/process-image/")
async def process_image(request: ImageProcessingRequest):
    """Process an image based on the requested operation."""
    try:
        # Asynchronously download the image
        image = await download_image(request.image_url)

        # Perform the requested operation
        if request.operation == "resize":
            image = resize_image(image, request.width, request.height)
        elif request.operation == "grayscale":
            image = ImageOps.grayscale(image)
        elif request.operation == "blur":
            image = image.filter(ImageFilter.BLUR)
        elif request.operation == "remove_background":
            image = remove_background(image)
        elif request.operation == "invert":
            image = ImageOps.invert(image)
        elif request.operation == "thumbnail":
            # Define the standard thumbnail size
            thumbnail_size = (150, 150)

            # Create the thumbnail
            image.thumbnail(thumbnail_size)
        else:
            raise HTTPException(status_code=400, detail="Unsupported operation.")

        # Save the processed image to a BytesIO object
        output = BytesIO()
        image.save(output, format="PNG")
        output.seek(0)

        # Generate a unique filename with a timestamp
        timestamp = int(time.time())
        filename = f"{request.operation}_image_{timestamp}.png"

        # Upload the image to the bucket
        image_url = upload_to_bucket(output, filename)

        return {
            "message": f"{request.operation.capitalize()} operation completed and uploaded successfully.",
            "image_url": image_url  # Return the URL of the uploaded image
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
Image Processing API Documentation
Setup and Deployment
Build and run the Docker container with:
bashdocker build -t image-processing-api .
docker run -p 8000:8000 image-processing-api
API Usage
The API processes images according to specified operations. All requests should be sent as JSON to the appropriate endpoint.

Example request payloads:

{
  "image_url": "https://example.com/sample-image.jpg",
  "operation": "grayscale"
}

{
  "image_url": "https://example.com/sample-image.jpg",
  "operation": "blur"
}

{
  "image_url": "https://example.com/sample-image.jpg",
  "operation": "invert"
}

{
  "image_url": "https://example.com/sample-image.jpg",
  "operation": "resize",
  "width": 300,
  "height": 200
}
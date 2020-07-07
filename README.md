# perspective_corrector

Perspective correction of documents with OpenCV.

Listens per default at port 8080

## Provides
    [GET/POST] /status
    [POST] /api/image/transform
    
  
## Usage

### Spring Rest
    File imgPath = new File("/path/to/image.jpg");
    final byte[] data = getImageBytearray(imgPath);

    MultiValueMap<String, String> fileMap = new LinkedMultiValueMap<>();
    ContentDisposition contentDisposition = ContentDisposition.builder("form-data").name(imgPath.getName()).filename(imgPath.getName()).build();
    fileMap.add(HttpHeaders.CONTENT_DISPOSITION, contentDisposition.toString());
    fileMap.add(HttpHeaders.CONTENT_TYPE, MediaType.IMAGE_JPEG_VALUE);
    
    HttpEntity<byte[]> fileEntity = new HttpEntity<>(data, fileMap);
    MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
    body.add("file", fileEntity);
    
    HttpHeaders headers = new HttpHeaders();
    headers.setContentType(MediaType.MULTIPART_FORM_DATA);
    headers.set("Accept", "*/*");
    
    HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, data, headers));
    RestTemplate template = new RestTemplate();
    template.getMessageConverters().add(new ResourceHttpMessageConverter());
    final ResponseEntity<Byte[]> entity = template.postForEntity("http://server.local:8080/api/image/transform", requestEntity, Byte[].class);

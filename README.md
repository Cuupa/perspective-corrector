# perspective_corrector

Perspective correction of documents with OpenCV.

Listens per default at port 5000

## Arguments
    -port [portnumber]

## Provides
    [GET/POST] /api/status
    [POST] /api/image/transform
    
  
## Usage

### Spring Rest
    final byte[] payload = readFile();

    RestTemplate upload = new RestTemplate();
    HttpHeaders headers = new HttpHeaders();
    headers.set("Content-Type", "multipart/form-data");
    headers.set("Accept", "*/*");

    MultiValueMap<String, String> fileMap = new LinkedMultiValueMap<>();
    ContentDisposition contentDisposition = ContentDisposition.builder("form-data")
        .name("key")
        .filename("perspective.png")
        .build();

    fileMap.add("Content-Disposition", contentDisposition.toString());
    fileMap.add("Content-Type", "image/jpeg");

    HttpEntity<byte[]> entity = new HttpEntity<>(payload, fileMap);
    MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
    body.add("file", entity);
    HttpEntity<MultiValueMap<String, Object>> requestData = new HttpEntity<>(body, headers);

    ResponseEntity<byte[]> responseEntity = upload.postForEntity("http://127.0.0.1:5000/api/image/transform", requestData, byte[].class);

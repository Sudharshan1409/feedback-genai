function aws-lambda-update() {
    echo "Updating lambda code"
    rm -rf lambda_function lambda_function.zip
    mkdir lambda_function
    cp ./main.py ./lambda_function/lambda_function.py
    cp ./data.json ./lambda_function/data.json
    pip3 install -r requirements.txt -t ./lambda_function
    cd lambda_function
    zip -r9 lambda_function.zip .
    mv lambda_function.zip ..
    cd ..
    aws lambda update-function-code \
        --function-name feedback-test \
        --zip-file fileb://lambda_function.zip
}


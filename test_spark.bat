@echo off
echo ========================================
echo Spark 2.0 Test Server
echo ========================================
echo.
echo Starting HTTP server on http://localhost:8080
echo Open http://localhost:8080/test_spark.html in your browser
echo.
echo Press Ctrl+C to stop the server
echo.

python -m http.server 8080
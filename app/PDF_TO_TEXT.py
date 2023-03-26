import PyPDF2

#import the Python library PyPDF2 , which will take care of adding text, and manipulating pdfs
#To convert the pdf file to text, we will first open the file using the open() function in the “rb” mode. i.e.Instead of the file contents,
# we will read the file in binary mode. The open() function takes the filename as the first input argument and the mode as the second input argument. After opening the file,
#  it returns a file object that we assign to the variable myFile. 
# After getting the file object, we will create a pdfFileReader object using the PdfFileReader() function defined in the PyPDF2 module. 
# The PdfFileReader() function accepts the file object containing the pdf file as the input argument and,
#  returns a pdfFileReader object. Using the pdfFileReader, we can convert the PDF file to text.


myFile = open("/Users/mirmacair/Desktop/HACKATHON_worktern/sample.pdf", "rb")
output_file = open("USER'S RESUME.txt", "w")
pdfReader = PyPDF2.PdfFileReader(myFile)
numOfPages = pdfReader.numPages
print("The number of pages in the pdf file is:", numOfPages)

#After getting the number of pages in the PDF file, we will use a for loop to process all the pages of the pdf file. In the for loop, we will extract each page from the PDF file using the getPage() method. The getPage() method, when invoked on a pdfFileReader object, accepts the page number 
# as an input argument and returns a pageObject containing data from the specified page of the PDF file.

#After getting the pageObject, we will use the extractText() method to extract text from the current page. After that, we will write the extracted text to the output text file.

#After extracting the text from all the pages in pdf, we will close both the text file and the pdf file. Otherwise, the changes will not be saved.


for i in range(numOfPages):
    page = pdfReader.getPage(i)
    text = page.extractText()
    output_file.write(text)
output_file.close() #closing the output file
myFile.close() #closing the file
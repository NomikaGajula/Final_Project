
const form = document.forms['contact-form'];

// Function to handle form submission and data sending
function submitFormAndOpenRoute() {
  // URL for your Google Sheets script
  const scriptURL = 'https://script.google.com/macros/s/AKfycbx_gJ_3FammuefwzOEHN9Y1KsQkOf9ZTXLmV3Bg13RlfGyHQdQ7tvBDfoN-ldjPE1R89Q/exec';

  // URL for your Flask route
  const flaskURL = 'http://127.0.0.1:5000/run_script';

  // Use the FormData from the form
  const formData = new FormData(form);

  // Send data to Google Sheets..............................
  fetch(scriptURL, { method: 'POST', body: formData })
    .then(response => response.json())  // Assuming the response is in JSON format
    .then(data => console.log('Data sent to Google Sheets:', data))
    .catch(error => console.error('Error sending data to Google Sheets:', error));

  // Send data to Flask route
  fetch(flaskURL, { method: 'POST', body: formData })
  // .then(data => console.log('This is working', data))
    .then(response => response.text())
    .then( 
      alert('Thank you! Your form is submitted successfully and data sent to Google Sheets'))
      .then(() => { window.location.reload(); })
    .catch(error => console.error('Error sending data to Flask route:', error));

}

// Event listener for the "Submit" button
form.addEventListener('submit', e => {
  e.preventDefault();
  submitFormAndOpenRoute();
});



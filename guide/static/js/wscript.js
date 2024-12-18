const url = 'https://api.openweathermap.org/data/2.5/weather';
const apiKey = '6ace5ce2f4ed32be547383197ca8eac1';

$(document).ready(function () {
    // Set up an event listener for the Enter key
    $('#city-input').on('keydown', function (e) {
        if (e.key === 'Enter') {
            let city = $(this).val();
            if (city) {
                weatherFn(city);
            }
        }
    });
    // Default city on load
    weatherFn('Mumbai');
});

async function weatherFn(cName) {
    const temp = `${url}?q=${cName}&appid=${apiKey}&units=metric`;
    try {
        const res = await fetch(temp);
        const data = await res.json();
        if (res.ok) {
            weatherShowFn(data);
        } else {
            alert('City not found. Please try again.');
        }
    } catch (error) {
        console.error('Error fetching weather data:', error);
    }
}

function weatherShowFn(data) {
    $('#city-name').text(data.name);
    $('#date').text(moment().format('MMMM Do YYYY, h:mm:ss a'));
    $('#temperature').html(`${data.main.temp}°C`);
    $('#description').text(data.weather[0].description);
    $('#wind-speed').html(`Wind Speed: ${data.wind.speed} m/s`);
    // Add the correct icon URL based on OpenWeatherMap icons
    $('#weather-icon').attr('src', `http://openweathermap.org/img/wn/${data.weather[0].icon}.png`);
    $('#weather-info').fadeIn();
}

// Scroll function to change navbar background on scroll
window.onscroll = function () {
    scrollFunction();
};

function scrollFunction() {
    var navbar = document.getElementById("navbar");
    if (document.body.scrollTop > 80 || document.documentElement.scrollTop > 80) {
        navbar.style.backgroundColor = "#fff";
        navbar.style.boxShadow = "0 2px 5px rgba(0, 0, 0, 0.2)";
    } else {
        navbar.style.backgroundColor = "rgba(255, 255, 255, 0.8)";
        navbar.style.boxShadow = "0 2px 5px rgba(0, 0, 0, 0.1)";
    }
}

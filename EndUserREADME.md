# How to use the STOFS Event Viewer/Analyzer Dashboard

The STOFS Event Viewer/Analyzer Dashboard allows the user to view STOFS model guidance output, including water level model guidance superposed on water level observations, for particular storm events in localized regions where they have impacts.

At the top left corner of the dashboard you will see the three-line menu icon.

<img width="348" height="254" alt="image" src="https://github.com/user-attachments/assets/1589369e-d9af-4545-b012-fdba1c4c76a3" />

If the Event Configuration Panel is not open, you will see the Map / Time series / Statistics / Scatter / Extremes menu at far left, as shown above. 

By clicking the three-line menu icon at top left corner, you can toggle whether the Event Configuration Panel is open or closed. When open, the panel appears to the left of the Map / Time series ... menu, as follows:

<img width="542" height="598" alt="image" src="https://github.com/user-attachments/assets/0115aa6c-4cc1-4b18-b467-517c27207b14" />

To start using the dashboard, first click on the pull-down Event menu, at the top of the Event Configuration Panel, select the storm of interest, and the click the blue "Load event" button:

<img width="307" height="191" alt="image" src="https://github.com/user-attachments/assets/33dbdc75-eb48-4bdf-83f5-f3f086530c61" />
<img width="316" height="137" alt="image" src="https://github.com/user-attachments/assets/49529667-5841-4216-9f4d-3e333d833ecc" />

Next, you will configure the parameter, the model outputs, the datum, and the statistical properties you want to view-- these choices will not take effect until you click the blue "Update plots" button at the bottom of the panel. 

1. Click the "Plot type" pull-down menu and select what you parameter want to view:

<img width="310" height="156" alt="image" src="https://github.com/user-attachments/assets/56731ad7-8a17-4cbd-b2a6-8c869f184e37" />

You can choose among these parameters:

 - Combined Water Level "CWL"-- the combined tidal, and storm-surge or weather-band, components
 - Pressure -- the surface air pressure
 - Wind -- Wind speed

2. Click the "Forecast type" pulldown menu and select among the nowcast and forecast choices:

<img width="311" height="495" alt="image" src="https://github.com/user-attachments/assets/ec6c832b-fd30-4aa3-a971-bdbb48df9513" />


All times are given in UTC. Information about STOFS nowcast and forecast cycles is explained at these links: 
- [STOFS 2D Global](https://noaa-gestofs-pds.s3.amazonaws.com/README.html) 
- [STOFS 3D Atlantic](https://noaa-nos-stofs3d-pds.s3.amazonaws.com/README_Atl.html)
- [STOFS 3D Pacific](https://noaa-nos-stofs3d-pds.s3.amazonaws.com/README_Pac.html)

For nowcast, the timeseries plots will show the sequential series of 6-hr periods ending at the time of each model-run date, throughout the full forecast extent relative to the storm initiation datetime (start date/time designated for the storm, for purposes of defining the event).

For forecasts, the timeseries plots will show the forecast starting at the time of the model-run and extending through the full forecast extent relative to the storm initiation datetime.



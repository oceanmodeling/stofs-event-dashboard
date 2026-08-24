# How to use the STOFS Event Viewer/Analyzer Dashboard

The STOFS Event Viewer/Analyzer Dashboard allows the user to view STOFS model guidance output together with observations, for particular storm events in localized regions where they have impacts.

This includes water level model guidance superposed on water level observations, in time series and scatterplots; tables of statistical measures of the model performance relative to the observations; and the day/time of observed peaks with corresponding modeled values.

There are two main stages for the user to proceed through, detailed in the two subsections that follow here: first, configuring the event, the models, and the parameters of interest; and second, viewing the resulting graphical and tabular comparison results.

## Configure the event, the model choices, and the parameters you want to view

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

For nowcast, the timeseries plots will show the sequential series of 6-hr periods ending at the time of each model-run date, throughout the full forecast extent relative to the storm initiation datetime (start date/time designated for the storm, for purposes of defining the event; this is the date/time of the first forecast in the list).

For forecasts, the timeseries plots will show the forecast starting at the time of the model-run and extending through the full forecast extent relative to the storm initiation datetime.

3. Click the "Station" pulldown menu and select the observations station to be treated

<img width="320" height="325" alt="image" src="https://github.com/user-attachments/assets/b787fe9e-19a9-4a05-ace1-527e87f9e71b" />

Once the station is selected you will see choices of model/parameter available, for that station:

<img width="301" height="114" alt="image" src="https://github.com/user-attachments/assets/7d8cde45-1286-4abd-9fe8-0faafc9eb9de" />

Each model/parameter in this list is a button that can be toggled by clicking it. Toggle on all the items in the list you want to be included in plots you view. 


(in this image, the model/parameter combinations `('stofs_2d_glo','cwl_bias_corrected')` and `('stofs_2d_glo','cwl_raw')` are selected)

Note: "cwl_bias_corrected" is a bias-corrected combined water level product using the prior 5-day period of observations, which is documented [here](https://repository.library.noaa.gov/view/noaa/72262 ); "cwl_raw" is the combined water level model output without the bias correction applied.

4. Click the "Datum" pulldown menu and select the vertical datum to be used for the viewer plots.

<img width="289" height="130" alt="image" src="https://github.com/user-attachments/assets/bc68c0c0-2328-4a31-a4fd-f0efb9af72e5" />

5. As desired, adjust the statistical properties to be used in the peaks calculations and plots: POT quantile (%) and POT Window (hours). POT stands for Peaks Over Thresholds. These will be annotated on some of the dashboard plots to be viewed.

6. Important:
 - Be sure to click on the blue "Update plots" button at the bottom of the Event Configuration Panel, when you have finished entering the information in #1 through #5 above. Otherwise your changes will not take effect. Changing the value in the pulldown menu alone does not cause the change to take effect, you must also click the "Update plots" button.
 - After you have viewed plots, come back and repeat the above steps, to choose a different combination of event, models, and/or parameters-- and then click the "Update plots" button again to see plots for the new combination.

## View the resulting plots

The menu to the right of the Event Configuration Panel includes a list of plot selections you can choose, by clicking the name. Each plot is interactive, with zoom/pan and the ability to save the plot to a file (see controls along the right edge of the plot), and information panels that appear on hovering over features in the plot.

1. Map

This shows the map of the event boundaries (default, from National Hurricane Center), superposed on a map with the locations of observational sites shown. You can hover over the observational stations to see their lat-lon, various IDs, and station type. These "Station" pulldown described above shows the station numbers visible on hovering over the stations on this map.

<img width="926" height="484" alt="image" src="https://github.com/user-attachments/assets/cde6166f-803b-43e1-8be0-04e3b12cec9b" />


2. Time series

The timeseries plot shows the period from the initiation date/time of the event through the ensuing full forecast period. It has a legend at the right, which explains the color-coded lines for model/parameter combinations (and "obs" for measurements from the station gauge), as well as the event peak symbol (denoted "obs extreme") for the Peaks Over Thresholds calculations. Again, zoom/pan and saving to file are available options, and hovering over the curves shows an info box with corresponding numerical values.

<img width="1429" height="284" alt="image" src="https://github.com/user-attachments/assets/49f7420a-e348-4dfa-b003-3c8e97e344b2" />

3. Statistics

This is a table of statistical values/metrics computed using the various sets of model and observation pairs, for each parameter.

<img width="1431" height="91" alt="image" src="https://github.com/user-attachments/assets/ce5ee7fc-0c33-4c13-bfeb-dc18866a2644" />

Units for bias, rms, and rmse correspond to what is shown on the axes of the Time series plot just described. 
- bias = mean bias, model relative to observations
- ? rms = unbiased root mean square error ?
- rmse = total root mean square error

Unitless parameters are:
- cr = pearson correlation coefficient, -1 to +1
- nse = Nash-Sutcliffe Efficiency (-inf to +1)
- kge = Kling-Gupta Efficiency (-inf to +1)
- R1 = ?
- R3 = ?
- error = ?

4. Scatter

This is a scatterplot of model vs observations for each parameter with a best-fit line.

<img width="1427" height="369" alt="image" src="https://github.com/user-attachments/assets/44b2f014-d999-4027-9714-6b4a9c6406c2" />

Percentiles are ??

5. Extremes

This table shows the date/time identified as the peak, using the observations, together with the corresponding results from the models.

<img width="745" height="81" alt="image" src="https://github.com/user-attachments/assets/dcc23c91-dd86-43bf-bf2d-7f5232038763" />

Units are same as in the Time series and Scatter plots.





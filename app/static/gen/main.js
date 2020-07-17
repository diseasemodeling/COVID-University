function makeMap(parentName, SVG_name, fromDate, colorBy, map_data, covid_data){
  d3.select("#" + SVG_name).remove()

  parentWidth = document.getElementById(parentName).offsetWidth
  var margin = {top: 10, right: 10, bottom: 10, left: 10},
  height = parentWidth - margin.top - margin.bottom
  width = parentWidth - margin.left - margin.right

  color = d3.scaleSequentialLog([1, d3.max(covid_data, d => d[colorBy])], d3.interpolateReds)

  var svg = d3.select("#" + parentName)
    .append("svg")
      .attr("id", SVG_name)
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .attr("viewBox", [0, 0, 975, 610])
      .on("click", reset)
    .append("g")
      .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

  const g = svg.append("g");

  path = d3.geoPath()

  data = {}
  covid_data_filter = covid_data.filter(d =>  d.date == fromDate)
  Object.entries(covid_data_filter).forEach(d => data[d[1].fips] = d[1][colorBy])


  svg.selectAll("dot")
      .data(data)
  .enter().append("circle")
      .attr("r", 5)
      .attr("cx", function(d) { return x(d.date); })
      .attr("cy", function(d) { return y(d.close); })
      .on("mouseover", function(d) {
          div.transition()
              .duration(200)
              .style("opacity", .9);
          div	.html(formatTime(d.date) + "<br/>"  + d.close)
              .style("left", (d3.event.pageX) + "px")
              .style("top", (d3.event.pageY - 28) + "px");
          })
      .on("mouseout", function(d) {
          div.transition()
              .duration(500)
              .style("opacity", 0);
      });

  g.append("g")
    .attr("fill", "#444")
    .attr("cursor", "pointer")
    .selectAll("path")
    .data(topojson.feature(map_data, map_data.objects.states).features)
    .join("path")
      .attr("fill", d =>  color(data[d.id]))
      .on('click', clicked)
      .attr("d", path)
    .append("title")
      .text(d => d.properties.name + " -- " + colorBy + " -> " + data[d.id]);

  const zoom = d3.zoom()
    .scaleExtent([1, 4])
    .on("zoom", zoomed);

  g.append("path")
    .attr("fill", "none")
    .attr("stroke", "white")
    .attr("stroke-linejoin", "round")
    .attr("d", path(topojson.mesh(map_data, map_data.objects.states, (a, b) => a !== b)));

  svg.call(zoom);

  function reset() {
    svg.transition().duration(750).call(
      zoom.transform,
      d3.zoomIdentity,
      d3.zoomTransform(svg.node()).invert([width / 2, height / 2])
    );
  }

  function clicked(d) {
    clickState = d.properties.name
    document.getElementById('selectState').value = clickState
    statistic = d3.select("#colorState")._groups[0][0].value
    updateMap()
  }

  function zoomed() {
    const {transform} = d3.event;
    g.attr("transform", transform);
    g.attr("stroke-width", 1 / transform.k);
  }

}



function dateToEight(date){
  year = String(date.getFullYear())
  month = String(date.getMonth() + 1)
  day = String(date.getDate())
  if (day.length == 1){
    day = '0' + day
  }
  if (month.length == 1){
    month = '0' + month
  }
  eightDate = year + month + day
  return eightDate
}

function eightToDate(eight){
  eight = String(eight)
  year = parseInt(eight.substring(0,4))
  month = parseInt(eight.substring(4,6))-1
  day = parseInt(eight.substring(6,8))
  date = new Date(year,month,day)
  return date
}

function weekOfYear(date) {
  // from date input
  //year = parseInt(date.substring(0,4))
  //month = parseInt(date.substring(5,7))-1
  //day = parseInt(date.substring(8,10))
  //date = new Date(year,month,day)
  // dayOfWeek = date.getDay()
  year= date.getFullYear()
  start = new Date(year, 0, 0)
  diff = date - start
  oneWeek = 1000 * 60 * 60 * 24 * 7
  week = Math.floor(diff / oneWeek)
  return week
}

function makeForesight(parentName, covid_data, state="New York", dataType='positive'){

  document.getElementById(parentName).innerHTML = ""
  // clears the data from the last time

  console.log(dataType)

  config = {
  pointType: 'mmwr-week', // Default is week
  axes: {
    y: {
      title: dataType // Title for the y axis
    },
    x: {
      title: 'Week of 2020'
    }
    },
  confidenceIntervals: ['50%', '75%'],
  }


  weekToEight = {}
  filtered = covid_data.filter(d => eightToDate(d.date).getDay() == 0)
  filtered = filtered.filter(d => d.state == states_hash[state])
  Object.entries(filtered).forEach(d => weekToEight[weekOfYear(eightToDate(d[1].date))] = d[1][dataType] )


  timePoints = [...Array(51).keys()].map(w => {
    return { week: w + 1, year: 2020 }
  })


  weekNow = weekOfYear(new Date())

  actual = []
  for (i = 0; i < 51; i++) {
    if (i in weekToEight){
      actual.push(weekToEight[i])
    }else{
      actual.push(null)
    }}

  // this is where the RL/simulations models comes in
  predictions = timePoints.map(tp => {
    if (tp.week >= weekNow) {
    // We only predict upto week 30
      return null
    } else {
      to_plot = []
      for (i = tp.week; i < weekNow; i++) {
        if (typeof weekToEight[i] == "number"){
          r = parseFloat(weekToEight[i])
          to_plot.push({point: r,
                        low: [Math.max(0, r - (r * .1)), Math.max(0, r - r * 0.2)],
                        high: [Math.max(0, r + (r * .1)), Math.max(0, r + (r * .2))]})
        } else {
          to_plot.push({point: parseFloat(0)})
        }
      }
      return {
        series: Array.from(to_plot)
      }
    }
  })

  data = {
    timePoints,
    actual: actual,
    models: [
      {
        id: 'actual tester',
        meta: {
          name: 'Name',
          description: 'Model description here',
          url: 'http://github.com'
        },
        pinned: false, // Setting true shows the model in top section of the legend
                       // In case of absence of `pinned` key (or false), the model
                       // goes in the bottom section
        predictions: predictions,
        style: { // Optional parameter for applying custom css on svg elements
          color: '#4682b4', // Defaults to values from the internal palette
          point: {
            // Style for the dots in prediction
          },
          area: {
            // Style for the confidence area (shaded region around the line)
          },
          line: {
            // Style for the main line
          }
        }
      }
    ]
  }

  timeChart = new d3Foresight.TimeChart('#' + parentName, config)
  timeChart.plot(data)
  timeChart.update(10)
  timeChart.moveForward()
  timeChart.moveBackward()

}

function makeCovidTrackerLine(parentName, SVG_name, covid_data,
                              scaleType='linear', state="New York",
                              dataType='positive', cumulative='cumulative',
                              simulation=null, simDataType=null, aspect=null){

  d3.select("#" + SVG_name).remove()
  // clears the data from the last time
  if (aspect == null){aspect = .33}
  aspectRatio = aspect

  title_mapping = {'positive':' Positive Cases - Cumulative',
                   'positiveIncrease': ' Positive Cases - Daily Increase',
                   'death':' Deaths - Cumulative',
                   'deathIncrease': ' Deaths - Daily Increase',
                   'hospitalized':' Hospitilizations - Cumulative',
                   'hospitalizedIncrease': ' Hospitilizations - Daily Increase',
                   'totalTestResults': ' Tests Taken - Cumulative',
                   'totalTestResultsIncrease': ' Tests Taken - Daily Increase',
                  }

  parentWidth = document.getElementById(parentName).offsetWidth
  var margin = {top: 30, right: 20, bottom: 20, left: 30},
  height = parentWidth * aspectRatio - margin.top// - margin.bottom
  width = parentWidth - margin.left - margin.right

  data = covid_data.filter(d => d.state == states_hash[state])

  if (cumulative == 'daily'){
    dataType = dataType + 'Increase'
  }

  if (simulation == null){
    x = d3.scaleUtc()
          .domain(d3.extent(data, d => eightToDate(d.date)))
          .range([margin.left, width - margin.right])
    if (scaleType == 'linear'){
      y = d3.scaleLinear()
            .domain([d3.min(data, d => d[dataType]), d3.max(data, d => d[dataType])])
            .range([height - margin.bottom, margin.top])
    }else{
      y = d3.scaleLog()
            .domain([1, d3.max(data, d => d[dataType])])
            .range([height - margin.bottom, margin.top])
    }
  }else{
    x = d3.scaleUtc()
          .domain(d3.extent(simulation, d => new Date(d.Date)))
          .range([margin.left, width - margin.right])

    if (scaleType == 'linear'){
      y = d3.scaleLinear()
            .domain([d3.min(simulation, d => parseFloat(d[simDataType])),
                     d3.max(simulation, d => parseFloat(d[simDataType]))])
            .range([height - margin.bottom, margin.top])
    }else{
      y = d3.scaleLog()
            .domain([1, d3.max(simulation, d => parseFloat(d[simDataType]))])
            .range([height - margin.bottom, margin.top])
    }
  }



  var svg = d3.select("#" + parentName)
    .append("svg")
    .attr("id", SVG_name)
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  svg.append("g")
     .attr("transform", `translate(0,${height - margin.bottom})`)
     .call(d3.axisBottom(x).ticks(10)) ;

  svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y)
            .ticks(5)
          );


  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left)
    .attr("x",0 - (height / 2))
    .attr("dy", "1em")
    .style("font-size", "8px")
    .style("text-anchor", "middle")
    .text("Value (in Thousands)");

    svg.append("g")
    .attr("stroke", "currentColor")
    .attr("stroke-opacity", 0.1)
    .call(g => g.append("g")
      .selectAll("line")
      .data(x.ticks())
      .join("line")
        .attr("x1", d => 0.5 + x(d))
        .attr("x2", d => 0.5 + x(d))
        .attr("y1", margin.top)
        .attr("y2", height - margin.bottom))
    .call(g => g.append("g")
      .selectAll("line")
      .data(y.ticks())
      .join("line")
        .attr("y1", d => 0.5 + y(d))
        .attr("y2", d => 0.5 + y(d))
        .attr("x1", margin.left)
        .attr("x2", width - margin.right));

  drawLine = d3.line()
               .y(function(d) {return y(d[dataType])})
               .x(function(d) {return x(eightToDate(d.date))})

       // Add the line
  svg.append("path")
     .datum(data)
     .attr('class', 'line')
     .attr("fill", "none")
     .attr("stroke", "steelblue")
     .attr("stroke-linejoin", "round")
     .attr("stroke-linecap", "round")
     .attr("stroke-width", 2.25)
     .attr("d", d => drawLine(d))

     if (simulation !== null){
       drawSimLine = d3.line()
                    .y(function(d) {return y(parseFloat(d[simDataType]))})
                    .x(function(d) {return x(new Date(d.Date))})
       svg.append("path")
         .datum(simulation)
         .attr('class', 'line')
         .attr("fill", "none")
         .attr("stroke", "red")
         .attr("stroke-width", 1.5)
         .attr("d", d => drawSimLine(d))
     }

  // add the title
  svg.append("text")
       .attr("font-family", "sans-serif")
       .attr("font-size", 18)
       .attr("x", (width / 2))
       .attr("y", 10 - (margin.top / 2))
       .attr("text-anchor", "middle")
       .text(title_mapping[dataType]);
}



function format_datetime(date){
  month = '' + (date.getMonth() + 1),
  day = '' + date.getDate(),
  year = date.getFullYear();
  if (month.length < 2)
    month = '0' + month;
  if (day.length < 2)
    day = '0' + day;
  date = [year, month, day].join('-');
  return date
}

function makeUserInputLine(parentName, SVG_name, input_data, dataType='Cases', aspect=null){

  d3.select("#" + SVG_name).remove()
  if (aspect == null){aspect = .33}
  aspectRatio = aspect
  // clears the data from the last time
  parentWidth = document.getElementById(parentName).offsetWidth
  var margin = {top: 25, right: 10, bottom: 20, left: 20},
  height = parentWidth * aspect - margin.top - margin.bottom
  width = parentWidth - margin.left - margin.right

  data = prep_policy_d3(input_data, dataType)


  x = d3.scaleLinear()
        .domain(d3.extent(data, d => d.x))
        .range([margin.left, width - margin.right])

  y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.y)])
        .range([height - margin.bottom, margin.top])

  var svg = d3.select("#" + parentName)
    .append("svg")
    .attr("id", SVG_name)
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var xAxis = svg.append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
     .call(d3.axisBottom(x));

  var yAxis= svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
     .call(d3.axisLeft(y).ticks(5));

  drawLine = d3.line()
               .y(function(d) {return y(d.y)})
               .x(function(d) {return x(d.x)});

  // Add the line
  svg.append("path")
     .datum(data)
     .attr('class', 'line')
     .attr("fill", "none")
     .attr("stroke", "steelblue")
     .attr("stroke-width", 1.5)
     .attr("d", d => drawLine(d))

  svg.append("text")
    .attr("x", (width / 2))
    .attr("y", 0 - (margin.top / 2))
    .attr("text-anchor", "middle")
    .style("font-size", "14px")
    .style("text-decoration", "underline")
    .text(dataType);
}



function prep_policy_d3(input_data, dataType){
  if (dataType=='Contact Reduction'){
    data = {'x':input_data['Startday(ContactReduction)'],
            'y':input_data['ContactReduction(0to1)']}
  }else if (dataType=='Contact Tracing'){
    data = {'x':input_data['Startday(ContactTracing)'],
            'y':input_data['Testsperday(ContactTracing)']}
  }else if (dataType=='Universal Testing'){
    data = {'x':input_data['Startday(UniversalTesting)'],
            'y':input_data['Testsperday(UniversalTracing)']}
  }

  data['x'] = data['x'].filter(function (value) {
      return !Number.isNaN(value);
  });
  data['y'] = data['y'].filter(function (value) {
      return !Number.isNaN(value);
  });
  num_entries = data.x.length

  new_data = []
  for (var i=1; i<num_entries; i++){
    span = data['x'][i] - data['x'][i-1]
    for (var d=0; d<span; d++){
      new_data[new_data.length] = {'x':new_data.length, 'y':data['y'][i-1]}
    }
  }
  return new_data
}

function makeABCLine(parentName, SVG_name, input_data, dataType='Cases', aspect=null){

  d3.select("#" + SVG_name).remove()
  if (aspect == null){aspect = .33}
  aspectRatio = aspect
  // clears the data from the last time
  parentWidth = document.getElementById(parentName).offsetWidth
  var margin = {top: 15, right: 10, bottom: 20, left: 20},
  height = parentWidth * aspect - margin.top - margin.bottom
  width = parentWidth - margin.left - margin.right

  data = prep_ABCpolicy_d3(input_data)

  x = d3.scaleLinear()
        .domain(d3.extent(data, d => d.x))
        .range([margin.left, width - margin.right])

  y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d3.max([d.yA,d.yB,d.yC]))])
        .range([height - margin.bottom, margin.top])

  var svg = d3.select("#" + parentName)
    .append("svg")
    .attr("id", SVG_name)
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var xAxis = svg.append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
     .call(d3.axisBottom(x))

  svg.append("text")
    .attr("transform", "translate(" + (width/2) + " ," +  (height + margin.top) + ")")
    .style("text-anchor", "middle")
    .style("font-size", "12px")
    .text("Days Since Start of Simulation");

  var yAxis= svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5));

    svg.append("g")
    .attr("stroke", "currentColor")
    .attr("stroke-opacity", 0.1)
    .call(g => g.append("g")
      .selectAll("line")
      .data(x.ticks())
      .join("line")
        .attr("x1", d => 0.5 + x(d))
        .attr("x2", d => 0.5 + x(d))
        .attr("y1", margin.top)
        .attr("y2", height - margin.bottom))
    .call(g => g.append("g")
      .selectAll("line")
      .data(y.ticks())
      .join("line")
        .attr("y1", d => 0.5 + y(d))
        .attr("y2", d => 0.5 + y(d))
        .attr("x1", margin.left)
        .attr("x2", width - margin.right));

  drawLineA = d3.line()
               .y(function(d) {return y(d.yA)})
               .x(function(d) {return x(d.x)});

  drawLineB = d3.line()
                .y(function(d) {return y(d.yB)})
                .x(function(d) {return x(d.x)});

  drawLineC = d3.line()
                .y(function(d) {return y(d.yC)})
                .x(function(d) {return x(d.x)});

  // Add the line
  svg.append("path")
     .datum(data)
     .attr('class', 'line')
     .attr("fill", "none")
     .attr("stroke", "steelblue")
     .attr("stroke-width", 1.5)
     .attr("d", d => drawLineA(d))

  svg.append("path")
    .datum(data)
    .attr('class', 'line')
    .attr("fill", "none")
    .attr("stroke", "red")
    .attr("stroke-width", 1.5)
    .attr("d", d => drawLineB(d))

  svg.append("path")
   .datum(data)
   .attr('class', 'line')
   .attr("fill", "none")
   .attr("stroke", "green")
   .attr("stroke-width", 1.5)
   .attr("d", d => drawLineC(d))


  title_mapping = {'CR':'Contact Rate',
                   'CT':'Contact Tracing & Testing (Proportion of Population)',
                   'UT':'Mass Testing (Proportion of Population)'}
  svg.append("text")
    .attr("x", (width / 2))
    .attr("y", (margin.top-10))
    .attr("text-anchor", "middle")
    .style("font-size", "14px")
    .style("text-decoration", "underline")
    .text(title_mapping[dataType]);

    rightSide = parentWidth - margin.left - margin.right - 40
    middleHeight = height / 2
    // Handmade legend
    svg.append("circle").attr("cx",rightSide).attr("cy",middleHeight - 10).attr("r", 4).style("fill", "steelblue")
    svg.append("circle").attr("cx",rightSide).attr("cy",middleHeight).attr("r", 4).style("fill", "red")
    svg.append("circle").attr("cx",rightSide).attr("cy",middleHeight + 10).attr("r", 4).style("fill", "green")
    svg.append("text").attr("x", rightSide+10).attr("y", middleHeight - 10).text("Plan A").style("font-size", "10px").attr("alignment-baseline","middle")
    svg.append("text").attr("x", rightSide+10).attr("y", middleHeight).text("Plan B").style("font-size", "10px").attr("alignment-baseline","middle")
    svg.append("text").attr("x", rightSide+10).attr("y", middleHeight + 10).text("Plan C").style("font-size", "10px").attr("alignment-baseline","middle")
}



function prep_ABCpolicy_d3(input_data, dataType){
  data = {'x':input_data['Date'],
          'yA':input_data['PlanA'],
          'yB':input_data['PlanB'],
          'yC':input_data['PlanC'],
          }
  num_entries = data.x.length
  new_data = []
  for (var i=1; i<num_entries; i++){
    span = 7
    for (var d=0; d<span; d++){
      new_data[new_data.length] = {'x':new_data.length, 'yA':data['yA'][i-1],
                                   'yB':data['yB'][i-1], 'yC':data['yC'][i-1]}
    }
  }
  return new_data
}

function tableToJson(id, type='index') {
  var table = document.getElementById(id)
  var data = [];
  // first row needs to be headers
  var headers = [];
  for (var i=0; i<table.rows[0].cells.length; i++) {
      headers[i] = table.rows[0].cells[i].innerHTML.replace(/ /gi,'');
  }
  if (type == 'index'){
    // go through cells
  	start = 1
    for (var i=start; i<table.rows.length; i++) {
        var tableRow = table.rows[i];
        var rowData = {};
        for (var j=0; j<tableRow.cells.length ; j++) {
            rowData[headers[j]] = tableRow.cells[j].innerHTML;
        }
        data.push(rowData);
    }
  }else{
    // go through cells
    for (var i=0; i<table.rows[0].cells.length; i++) {
      var colData = [];
      remember = parseFloat(0)
      for (var j=1; j<table.rows.length ; j++) {
        if (i==0){
          colData.push(parseFloat(table.rows[j].cells[i].innerHTML.split(" ")[1]));
        }else{
          if (table.rows[j].cells[i].innerHTML != ''){
            remember = table.rows[j].cells[i].innerHTML
            colData.push(parseFloat(table.rows[j].cells[i].innerHTML));
          } else {
            colData.push(parseFloat(remember));
          }
      }}
      data[headers[i]] = colData;
    }
  }
  return data;
}

function makeSimLine(parentName, SVG_name, dataType=null, simA=null, simB=null, simC=null){

  d3.select("#" + SVG_name).remove()
  // clears the data from the last time
  aspectRatio = .33

  parentWidth = document.getElementById(parentName).offsetWidth
  var margin = {top: 15, right: 20, bottom: 20, left: 30},
  height = parentWidth * aspectRatio - margin.top// - margin.bottom
  width = parentWidth - margin.left - margin.right

  if (simA != null){maxA = d3.max(simA, d => parseFloat(d[dataType]))}else{maxA=0}
  if (simB != null){maxB = d3.max(simB, d => parseFloat(d[dataType]))}else{maxB=0}
  if (simC != null){maxC = d3.max(simC, d => parseFloat(d[dataType]))}else{maxC=0}
  total_max = Math.max(maxA, maxB, maxC)
  //x = d3.scaleLinear()
  //      .domain(d3.extent(simA, d => parseInt(d.Date)))
  //      .range([margin.left, width - margin.right])
  x = d3.scaleUtc()
        .domain(d3.extent(simA, d => new Date(d.Date)))
        .range([margin.left, width - margin.right])
  y = d3.scaleLinear()
        .domain([0, total_max])
        .range([height - margin.bottom, margin.top])


  var svg = d3.select("#" + parentName)
    .append("svg")
    .attr("id", SVG_name)
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  svg.append("g")
     .attr("transform", `translate(0,${height - margin.bottom})`)
     .call(d3.axisBottom(x).ticks(5)) ;

  svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y)
            .ticks(5)
          );

  svg.append("g")
    .attr("stroke", "currentColor")
    .attr("stroke-opacity", 0.1)
    .call(g => g.append("g")
      .selectAll("line")
      .data(x.ticks())
      .join("line")
        .attr("x1", d => 0.5 + x(d))
        .attr("x2", d => 0.5 + x(d))
        .attr("y1", margin.top)
        .attr("y2", height - margin.bottom))
    .call(g => g.append("g")
      .selectAll("line")
      .data(y.ticks())
      .join("line")
        .attr("y1", d => 0.5 + y(d))
        .attr("y2", d => 0.5 + y(d))
        .attr("x1", margin.left)
        .attr("x2", width - margin.right));

    svg.append("text")
      .attr("transform", "translate(" + (width/2) + " ," +  (height + margin.top) + ")")
      .style("text-anchor", "middle")
      .style("font-size", "12px")
      .text("Date");

    if (simA !== null){
      drawSimLine = d3.line()
                     .y(function(d) {return y(parseFloat(d[dataType]))})
                     .x(function(d) {return x(new Date(d.Date))})
                     //.x(function(d) {return x(parseInt(d.Date))})

    const pathA = svg.append("path")
      .datum(simA)
      .attr('class', 'line')
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1.5)
      .attr("d", d => drawSimLine(d))
     }
     if (simB !== null){
       drawSimLine = d3.line()
                       .y(function(d) {return y(parseFloat(d[dataType]))})
                       .x(function(d) {return x(new Date(d.Date))})
                       //.x(function(d) {return x(parseInt(d.Date))})

       const pathB = svg.append("path")
         .datum(simB)
         .attr('class', 'line')
         .attr("fill", "none")
         .attr("stroke", "red")
         .attr("stroke-width", 1.5)
         .attr("d", d => drawSimLine(d))

     }
     if (simC !== null){
       drawSimLine = d3.line()
                       .y(function(d) {return y(parseFloat(d[dataType]))})
                       .x(function(d) {return x(new Date(d.Date))})
                       //.x(function(d) {return x(parseInt(d.Date))})

       const pathC = svg.append("path")
         .datum(simC)
         .attr('class', 'line')
         .attr("fill", "none")
         .attr("stroke", "green")
         .attr("stroke-width", 1.5)
         .attr("d", d => drawSimLine(d))
     }

    if (dataType == 'Number of quarantined (per day)(with 0% false positives)'){
      title = 'Number of quarantines (onlty true positives)'
    } else if (dataType == 'Cumulative cost'){
      title = 'Cumulative cost ($, Millions)'
    } else {
      title = dataType
    }

  // add the title
  svg.append("text")
       .attr("font-family", "sans-serif")
       .attr("font-size", 14)
       .attr("x", (width / 2))
       .attr("y", 10 - (margin.top / 2))
       .attr("text-anchor", "middle")
       .text(title);
}



function format_datetime(date){
  month = '' + (date.getMonth() + 1),
  day = '' + date.getDate(),
  year = date.getFullYear();
  if (month.length < 2)
    month = '0' + month;
  if (day.length < 2)
    day = '0' + day;
  date = [year, month, day].join('-');
  return date
}

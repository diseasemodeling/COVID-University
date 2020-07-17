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

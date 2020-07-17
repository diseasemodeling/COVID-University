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

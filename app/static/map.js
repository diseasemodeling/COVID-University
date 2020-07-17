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

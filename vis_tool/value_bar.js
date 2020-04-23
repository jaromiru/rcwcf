val_bar = function(value) {
  // set the dimensions and margins of the graph
  var margin = {top: 5, right: 0, bottom: 40, left: 40},
      width = 50 - margin.left - margin.right,
      height = 70 - margin.top - margin.bottom;

  // append the svg object to the body of the page
  // memory leak!
  var svg = d3
    .create("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)

  svg_group = svg.append("g")
      .attr("transform",
            "translate(" +(0 + margin.left) + "," + (295 + margin.top) + ")");

    // Add X axis
    var x = d3.scaleBand()
      .domain([0])
      .range([0, width]);

    // Y axis
    var y = d3.scaleLinear()
      .domain([0, 1])
      .range([ height, 0 ])
      // .padding(.1);

    //Bars
    svg_group
      .append("rect")
      .attr("x", x(0) )
      .attr("y", y(value) + 1 )
      .attr("width", x.bandwidth())
      .attr("height", y(0) - y(value) )
      .attr("fill", "gray")

    svg_group.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x).tickSize(0).tickFormat(''))

    svg_group.append("text")
      .attr("transform", `translate(15 , ${height/2})`)
      .text("V(s)");

    svg_group.append("g")
      .call(d3.axisLeft(y).ticks(2))
      .attr("font-size", null)
      .attr("font-family", null)

  return svg_group.node()
}
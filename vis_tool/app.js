accessor_f = function(x) {
  if ("children" in x){
    children = x.children

    // MAX_LEN = 4
    // if (children.length > MAX_LEN){
    //   children = children.slice(0, MAX_LEN)
    //   children.push({name: '...'})
    // }

    return children
  }

  if(typeof(x.prob) === 'undefined'){ // this is the root
    x.prob = 1.0
    x.parent_prob = 1.0
    x.selected = true
  }

  children = []
  for (prop in x){
    if (["open", "prob", "selected", "parent_prob"].indexOf(prop) >= 0){
      continue
    }

    item = x[prop]

    child = {name: prop, mask: item.mask, prob: item.prob * x.parent_prob, selected: item.selected, children: []}
    child.open = item.mask

    if (Array.isArray(item.value)){
      if (item.value.length > 0){
        child.children = item.value
        child.open = true

        if (!item.value.flag){
          item.value.flag = true

          for (v of item.value){
            total_p = 0.
            selected = false

            for (p in v){
              total_p += v[p].prob
              selected |= v[p].selected
            }

            // v.prob = total_p
            v.prob = total_p * item.prob
            v.parent_prob = item.prob

            v.selected = selected
          }
        }

        // child.children.open = truedsad
        // child.children.prob = total_p
        // child.children.selected = selected

        // console.log(child)
      }
    }
    else{
      text = item.value

      if (typeof(text) === "string"){
        if (text.startsWith("<p>")){
          text = text.substring(3)
        }
        if (text.length > 20){
          text = text.substring(0, 20) + "..."
        }
        text = "'" + text + "'"
      }

      child.value = text
    }

    children.push(child)
  }

  x.open = true
  return children
}

var svg = null
var root = null
var update = null

saveSVG = function(fname='result.svg', i=0){
  console.log("Downloading "+fname)
  var x = serialize(vizNode.node())
  // console.log(x)
  setTimeout(function(){download(x, fname)}, 250*i)
}

closeAll = function(){
  root.descendants().forEach((d, i) => {
    d.children = null;
    d.open = false;
  });
  update(root)
}

openAll = function(){
  root.descendants().forEach((d, i) => {
    d.children = d._children;
    d.open = true;
  });
  update(root)
}

vizNode = d3.select("#viz")
function update_viz(item){
  ret = chart(item.sample, accessor_f);

  svg = ret[0]
  root = ret[1]
  update = ret[2]

  vizNode.selectAll("g").remove()
  vizNode.selectAll("text").remove()

  vizNode.append(() => svg)
  vizNode.append(() => bar_chart(item.cls_probs, item.true_y))
  vizNode.append(() => val_bar(item.s_value))
  vizNode.append('text')
    .attr("transform", `translate(20 , 405)`)
    .text(`total cost: ${item.total_cost.toFixed(1)}`);

  // console.log(item)
}

var app = new Vue({
  el: '#app',
  data: {
    sample_id: 0,
    step_id: 0,
    max_steps: '?',
    dataset_name: dataset_name,
    data: data,
    current_sample: null,
    current_item: null
  },
  methods: {
    selectSample: function(){
      this.current_sample = this.data[this.sample_id]
      this.step_id = 0
      this.selectStep()
    },
    selectStep: function(){
        this.current_item = this.current_sample[this.step_id]
        update_viz(this.current_item)
    },
    nextStep: function(){
      this.step_id = Math.min(this.step_id + 1, this.current_sample.length - 1)
      this.selectStep()      
    },
    saveSample: function(){
      for(var i = 0; i < this.current_sample.length; i++){
        this.step_id = i
        this.selectStep();
        saveSVG(`${this.dataset_name}_smpl_${this.sample_id}_step_${i}.svg`, i);
      }
    },
  },
  filters: {
    to2digit: function (value) {
      if (Array.isArray(value)){
        return value.map(x => x.toFixed(2))
      }
      else {
        return value.toFixed(2);
      }
    }
  },
  created: function(){
    this.selectSample()
  }
})

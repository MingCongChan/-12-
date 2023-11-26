var express = require('express');
var router = express.Router();

const Busboy = require('busboy')
const path = require('path')
const fs = require('fs')

const { execSync, exec } = require('child_process')
/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});


router.get('/main',function (req,res){
  res.render('users',{
    layout:false,
      title: 'main',
      mainInfo: 'main paper'
    });
})

router.post('/upload', function (req, res) {
  // console.log(req.headers)
  let busboy = Busboy({ headers: req.headers });

  let data = {
    filename:'', // 图片名字
    encoding:'', // 图片大小
    mimeType:'', // 图片格式
    imgUrl:'' // 图片地址
  }

  let tmp = [];

  busboy.on('file', (name, file, info) => {
    const { filename, encoding, mimeType } = info;
    // 根据时间创建名字 - 防止重名
    const filePath = new Date().getTime() + path.extname(filename)
    // 保存数据
    data = {...info,filename:filePath}
    tmp.push(data);
    // 拼接地址
    const saveTo = path.join(__dirname, '../public/upload', filePath);
    // 写入流
    file.pipe(fs.createWriteStream(saveTo));
  });

  busboy.on('finish', function () {
    data.imgUrl = 'http://127.0.0.1:3000/uploads/' + data.filename
    let lpath = "E:\\Software\\tools\\anaconda3\\envs\\test\\python.exe" + " "+".\\public\\python\\demo.py " +__dirname+"\\..\\public\\upload\\" + data.filename
    console.log(lpath);
    // 返回图片
    exec(lpath, function(error,stdout,stderr){
    const filePath = path.resolve(__dirname,'../res/2.jpeg');
    console.log(stdout);
    let tmp = NaN;
    fs.readFile(filePath,function(err,data){
      tmp = Buffer.from(data, 'binary').toString('base64');
      res.render('users',{
        layout:false,
          title: 'image',
          mainInfo: 'demo',
          imgUrl:'data:image/jpeg;base64,'+tmp,//'/res/1700371949648.png',
          txt:"ok"+stdout+error+stderr
        });
    });
      // res.end("ok"+stdout+error+stderr);
    })

    //格式必须为 binary，否则会出错
    // 创建文件可读流
    // const cs = fs.createReadStream(filePath);
    // let tmp = NaN;
    // cs.on("data", chunk => {
    //     // console.log(chunk);
    //     tmp +=chunk;
    //     // res.write(chunk);
    // })
    // cs.on("end", () => {
    //     let ttt = Buffer.from(tmp, 'binary').toString('base64')
    //     // console.log(tmp);
    //   res.render('users',{
    //   layout:false,
    //     title: 'main',
    //     mainInfo: 'main paper',
    //     imgUrl:'data:image/jpeg;base64,'+ttt//'/res/1700371949648.png',
    //   });
    // })

  });
  return req.pipe(busboy);
})

module.exports = router;

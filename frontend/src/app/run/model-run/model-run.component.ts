import { Component, OnInit } from '@angular/core';
import { GridDataResult } from '@progress/kendo-angular-grid';
import { orderBy, SortDescriptor, State, process } from '@progress/kendo-data-query';
import { AppConsts } from 'src/app/services/app.consts';
import { AppService } from 'src/app/services/app.service';

@Component({
  selector: 'app-model-run',
  templateUrl: './model-run.component.html',
  styleUrls: ['./model-run.component.css']
})
export class ModelRunComponent implements OnInit {
  public classOption: any;
  public objectOption: any;
  uploadSaveUrl = this.appConsts.fileSaveUrl;
  freeText = ''
  // public classOption: any;
  // public classOption: any;
  public listClass: Array<string> = [
    "crash",
    "animals",
    "fire",
    "flooding",
    "landslide",
    "snow",
    "treefall",
    "collapse"
  ];
  public listObject: Array<string> = [
    "car",
    "person",
    "bicycle",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign"
  ];
  public listPosition: Array<string> = [
    "in front of"
  ]
  public listLocation: Array<string> = [
    "in city",
    "on highway"
  ]
  public listAction: Array<string> = [
    "go ahead",
    "collide"
  ]
  public listAttribute: Array<string> = [
    "gray car",
    "white car"
  ]
  public list
  dataGrids = [];
  dataGridsVid = []
  dataGridGridDataResult: GridDataResult;
  dataGridGridDataResultVid: GridDataResult;
  dataGridSortByField: SortDescriptor[] = [];
  dataGridSkip = this.appConsts.pageSkip;
  dataGridPageSize = this.appConsts.pageSize;
  public dataGridState: State = {
    skip: this.dataGridSkip,
    take: this.dataGridSkip + this.dataGridPageSize,
    filter: {
      logic: 'and',
      filters: []
    }
  };
  apiRoot = null;
  constructor(
    private appService: AppService,
    public appConsts: AppConsts
  ) {
    this.apiRoot = appService.apiRoot;
    // this.testAPI();
    // this.inccidents_classification();
   }

  ngOnInit() {
  }

  binddataItems() {
    // this.dataGridGridDataResult = {
    //   data: orderBy(this.dataGrids, this.dataGridSortByField),
    //   total: this.dataGrids.length
    // };

    this.dataGridGridDataResult = process(this.dataGrids, this.dataGridState);
    console.log(this.dataGridGridDataResult)
  }

  binddataItemsVid() {
    // this.dataGridGridDataResult = {
    //   data: orderBy(this.dataGrids, this.dataGridSortByField),
    //   total: this.dataGrids.length
    // };

    this.dataGridGridDataResultVid = process(this.dataGridsVid, this.dataGridState);
    console.log(this.dataGridGridDataResult)
  }

  async testAPI(){
    const data = await this.appService.doGET('test', null)
    console.log(data.Data)
  }

  async inccidents_classification(){
    const result = await this.appService.doGET('model-run/uploads', null)
    console.log(result.Data)
    console.log(result.DataVid)
    this.dataGrids = result.Data;
    this.dataGridsVid = result.DataVid
    this.binddataItems();
    this.binddataItemsVid();
  }

  async onGet(){
    console.log(this.classOption, this.objectOption)
    const result = await this.appService.doGET('detail-search/' + this.classOption, null)
    console.log(result.Data)
    
    this.dataGrids = result.Data;
    this.binddataItems();
  }

  async onclickFree(){
    const result = await this.appService.doGET('/STS/' + this.freeText, null)
    console.log(result.Data)
    this.dataGrids = result.Data;
    this.binddataItems();
  }

}

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Location {
  locations: string[];
}

export interface PricePrediction {
  estimated_price: number;
}

@Injectable({
  providedIn: 'root'
})
export class PricePredictionService {
  private apiUrl = 'https://realestatepriceprediction-v1.onrender.com'; // Change this to your deployed backend URL

  constructor(private http: HttpClient) { }

  getLocations(): Observable<Location> {
    return this.http.get<Location>(`${this.apiUrl}/get_location_names`);
  }

  predictPrice(totalSqft: number, location: string, bhk: number, bath: number): Observable<PricePrediction> {
    const formData = new FormData();
    formData.append('total_sqft', totalSqft.toString());
    formData.append('location', location);
    formData.append('bhk', bhk.toString());
    formData.append('bath', bath.toString());
    
    return this.http.post<PricePrediction>(`${this.apiUrl}/predict_home_price`, formData);
  }
} 